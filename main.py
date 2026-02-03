"""
MOMENTUM-X CLI Entrypoint

### ARCHITECTURAL CONTEXT
Node ID: cli.main
Graph Link: docs/memory/graph_state.json → "cli.main"

### PURPOSE
The operational command center for running Momentum-X.
Supports modes: scan (pre-market only), evaluate (scan + analyze), paper (full loop), backtest.

### USAGE
  python -m main scan          # Pre-market gap scanner only
  python -m main evaluate      # Scan + agent evaluation pipeline
  python -m main paper         # Full paper trading loop
  python -m main backtest      # CPCV backtest on historical data

### CRITICAL INVARIANTS
1. Default mode is always paper (INV-007).
2. Live mode requires explicit --live flag + confirmation prompt.
3. All runs log to structured JSON for post-analysis.
4. Ctrl+C graceful shutdown with position report.

Ref: SYSTEM_ARCHITECTURE.md (4 Market Phases)
Ref: ADR-004 (Rate Limiting)
Ref: DATA-001-EXT CONSTRAINT-010 (GCP us-east4 deployment)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings
from src.analysis.post_trade import PostTradeAnalyzer, TradeResult
from src.agents.prompt_arena import PromptArena
from src.agents.default_variants import seed_default_variants

logger = logging.getLogger("momentum_x")


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


async def cmd_scan(settings: Settings, interval: int = 30, once: bool = False) -> None:
    """
    Run pre-market gap scanner with optional continuous polling.

    Modes:
      --once: single scan iteration, print candidates, exit
      default: continuous polling every --interval seconds

    Pipeline per iteration:
      1. Fetch market snapshots from Alpaca
      2. EMC conjunction filter (Gap%, RVOL, ATR)
      3. GEX enrichment (if options data available)
      4. GEX hard filter (reject extreme positive GEX)
      5. Output ranked CandidateStock list

    Ref: SYSTEM_ARCHITECTURE.md (Phase 1: Pre-Market)
    Ref: ADR-014 (Pipeline Closure)
    """
    from src.core.scan_loop import ScanLoop
    from src.data.alpaca_client import AlpacaDataClient

    logger.info("═══ MOMENTUM-X PRE-MARKET SCANNER ═══")
    logger.info("Mode: %s | Feed: %s | Interval: %ds | Once: %s",
                settings.mode, settings.alpaca.feed, interval, once)

    client = AlpacaDataClient(settings.alpaca)

    # Verify connectivity
    try:
        account = await client.get_account()
        equity = float(account.get("equity", 0))
        logger.info("Account connected: equity=$%.2f", equity)
    except Exception as e:
        logger.error("Failed to connect to Alpaca: %s", e)
        logger.error("Check ALPACA_API_KEY and ALPACA_SECRET_KEY env vars")
        return

    scan_loop = ScanLoop(settings=settings)

    # Graceful shutdown
    shutdown = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info("Shutdown signal received.")
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    iteration = 0
    while not shutdown.is_set():
        iteration += 1
        now = datetime.now(timezone.utc)
        logger.info("─── Scan iteration #%d at %s ───", iteration, now.strftime("%H:%M:%S UTC"))

        try:
            # Fetch snapshots from Alpaca (most-active tickers)
            quotes = await _fetch_scan_quotes(client, settings)

            if quotes:
                candidates = scan_loop.run_single_scan(quotes)
                if candidates:
                    logger.info("Found %d candidates:", len(candidates))
                    for i, c in enumerate(candidates, 1):
                        logger.info(
                            "  %d. %s | Gap: %.1f%% | RVOL: %.1fx | Price: $%.2f%s",
                            i, c.ticker, c.gap_pct * 100, c.rvol, c.current_price,
                            f" | GEX_norm: {c.gex_normalized:.2f}" if c.gex_normalized is not None else "",
                        )
                else:
                    logger.info("No candidates passed filters this iteration.")
            else:
                logger.info("No market data available. Market may be closed.")

        except Exception as e:
            logger.error("Scan iteration failed: %s", e)

        if once:
            break

        # Wait for next interval or shutdown
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

    logger.info("Scanner stopped after %d iterations.", iteration)


async def _fetch_scan_quotes(client, settings: Settings) -> dict:
    """
    Fetch market snapshot data from Alpaca for scanner input.

    Returns dict of ticker → {current_price, previous_close, premarket_volume, ...}
    suitable for ScanLoop.run_single_scan().
    """
    try:
        # Use Alpaca most-active or screener endpoint if available
        # For now, fetch snapshots for a watchlist or most-active tickers
        tickers = await client.get_most_active_tickers(limit=50)
        if not tickers:
            return {}

        snapshots = await client.get_snapshots(tickers)
        quotes = {}
        for ticker, snap in snapshots.items():
            try:
                current = float(snap.get("latestTrade", {}).get("p", 0) or
                               snap.get("minuteBar", {}).get("c", 0))
                prev_close = float(snap.get("prevDailyBar", {}).get("c", 0))
                volume = int(snap.get("minuteBar", {}).get("v", 0) or
                            snap.get("dailyBar", {}).get("v", 0))
                avg_vol = max(1, int(snap.get("prevDailyBar", {}).get("v", 1)))

                if current > 0 and prev_close > 0:
                    quotes[ticker] = {
                        "current_price": current,
                        "previous_close": prev_close,
                        "premarket_volume": volume,
                        "avg_volume_at_time": avg_vol,
                        "float_shares": None,
                        "market_cap": None,
                        "has_news": True,  # Conservative: assume news may exist
                    }
            except (ValueError, TypeError, KeyError):
                continue

        return quotes

    except Exception as e:
        logger.warning("Failed to fetch scan quotes: %s", e)
        return {}


async def cmd_evaluate(settings: Settings) -> None:
    """
    Run scanner + full agent evaluation pipeline.
    Produces TradeVerdicts without executing orders.
    """
    from src.core.orchestrator import Orchestrator
    from src.data.alpaca_client import AlpacaDataClient
    from src.data.news_client import NewsClient

    logger.info("═══ MOMENTUM-X EVALUATE MODE ═══")
    logger.info("Mode: %s | Debate threshold: %.2f", settings.mode, settings.debate.mfcs_debate_threshold)

    orchestrator = Orchestrator(settings)

    # Example evaluation flow (requires live data subscription)
    logger.info("Orchestrator initialized with %d agents", 6)
    logger.info("Pipeline: Scanner → 5 Agents (parallel) → MFCS → Debate → Risk → Verdict")
    logger.info("Evaluation mode ready. Feed candidates via scan or manual ticker entry.")


async def cmd_paper(settings: Settings) -> None:
    """
    Full paper trading loop with ExecutionBridge wiring.

    Pipeline per cycle:
      1. ScanLoop → CandidateStock list
      2. Orchestrator.evaluate_candidates() → TradeVerdicts
      3. ExecutionBridge.execute_verdict(verdict, scored) → OrderResult
      4. PositionManager tracks lifecycle, circuit breaker, tranche exits
      5. On close: bridge.close_with_attribution() → Shapley → Elo

    Ref: SYSTEM_ARCHITECTURE.md (4 Market Phases)
    Ref: ADR-003 (Execution Layer)
    Ref: ADR-014 (Pipeline Closure)
    Ref: ADR-016 (Production Wiring)

    Phase 1: Pre-Market (4:00-9:30 ET) — scan, build ranked watchlist
    Phase 2: Market Open (9:30-10:00 ET) — evaluate top candidates, execute BUYs
    Phase 3: Intraday (10:00-15:45 ET) — monitor positions, tranche exits
    Phase 4: After-Hours (16:00+) — close all, Shapley, report
    """
    from src.core.orchestrator import Orchestrator
    from src.core.scan_loop import ScanLoop
    from src.data.alpaca_client import AlpacaDataClient
    from src.execution.alpaca_executor import AlpacaExecutor
    from src.execution.bridge import ExecutionBridge
    from src.execution.position_manager import PositionManager

    if settings.mode != "paper":
        logger.error("Paper command requires mode=paper. Current: %s", settings.mode)
        return

    logger.info("═══ MOMENTUM-X PAPER TRADING ═══")
    logger.info("Mode: PAPER | Max positions: %d | Max per position: %.1f%%",
                settings.execution.max_positions,
                settings.execution.max_position_pct * 100)
    logger.info("Stop-loss: %.1f%% | Daily loss limit: %.1f%%",
                settings.execution.stop_loss_pct * 100,
                settings.execution.daily_loss_limit_pct * 100)

    # ── Initialize pipeline components ──
    client = AlpacaDataClient(settings.alpaca)

    try:
        account = await client.get_account()
        equity = float(account.get("equity", 0))
        logger.info("Account connected: equity=$%.2f", equity)
    except Exception as e:
        logger.error("Failed to connect to Alpaca: %s", e)
        return

    orchestrator = Orchestrator(settings)
    scan_loop = ScanLoop(settings=settings)
    executor = AlpacaExecutor(config=settings.execution, client=client)
    position_manager = PositionManager(
        config=settings.execution,
        starting_equity=equity,
    )
    bridge = ExecutionBridge(
        executor=executor,
        position_manager=position_manager,
    )

    # Track candidates and their ScoredCandidates for execution
    watchlist: list = []  # CandidateStock from Phase 1
    session_trades: int = 0

    # Graceful shutdown handler
    shutdown = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info("Shutdown signal received. Closing positions...")
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Paper trading loop started. Press Ctrl+C to stop.")

    # Main loop — runs until shutdown
    while not shutdown.is_set():
        now = datetime.now(timezone.utc)
        hour_et = (now.hour - 5) % 24  # Approximate ET offset

        try:
            if 4 <= hour_et < 9:
                # ── Phase 1: Pre-Market Scanning ──
                logger.info("[Phase 1] Pre-Market scanning...")
                quotes = await _fetch_scan_quotes(client, settings)
                if quotes:
                    watchlist = scan_loop.run_single_scan(quotes)
                    logger.info("Watchlist: %d candidates", len(watchlist))
                    for c in watchlist[:5]:
                        logger.info(
                            "  %s | Gap: %.1f%% | RVOL: %.1fx",
                            c.ticker, c.gap_pct * 100, c.rvol,
                        )

            elif 9 <= hour_et < 10:
                # ── Phase 2: Market Open — Evaluate + Execute ──
                logger.info("[Phase 2] Market Open — evaluating %d candidates...", len(watchlist))
                if watchlist and bridge.position_manager.can_enter_new_position():
                    verdicts = await orchestrator.evaluate_candidates(watchlist[:5])
                    for verdict in verdicts:
                        if verdict.action == "BUY":
                            # Retrieve scored from orchestrator (cached during eval)
                            order = await bridge.execute_verdict(verdict, scored=None)
                            if order is not None:
                                session_trades += 1
                                logger.info(
                                    "ORDER FILLED: %s qty=%d @ $%.2f (order_id=%s)",
                                    order.ticker, order.qty, order.submitted_price, order.order_id,
                                )

            elif 10 <= hour_et < 16:
                # ── Phase 3: Intraday Monitoring ──
                positions = bridge.position_manager.open_positions
                logger.info("[Phase 3] Intraday — monitoring %d positions", len(positions))
                for pos in positions:
                    logger.info(
                        "  %s: qty=%d, entry=$%.2f, stop=$%.2f",
                        pos.ticker, pos.remaining_qty, pos.entry_price, pos.stop_loss,
                    )

            else:
                # ── Phase 4: After-Hours ──
                positions = bridge.position_manager.open_positions
                if positions:
                    logger.info("[Phase 4] Closing %d remaining positions...", len(positions))
                    for pos in positions:
                        # Time stop: close all at market close
                        enriched = await bridge.close_with_attribution(
                            ticker=pos.ticker,
                            exit_price=pos.entry_price,  # Will be replaced by live price
                        )
                        if enriched:
                            logger.info(
                                "CLOSED %s: PnL=$%.2f",
                                pos.ticker, enriched.pnl,
                            )
                            # ── Shapley → Elo feedback (ADR-016 D1) ──
                            try:
                                from src.analysis.post_trade import PostTradeAnalyzer
                                from src.analysis.shapley import ShapleyAttributor
                                from src.agents.prompt_arena import PromptArena

                                arena_path = Path("data/arena_ratings.json")
                                if arena_path.exists():
                                    arena = PromptArena.load(str(arena_path))
                                else:
                                    from src.agents.prompt_arena import seed_default_variants
                                    arena = seed_default_variants()

                                analyzer = PostTradeAnalyzer(arena=arena)
                                attributor = ShapleyAttributor()
                                matchups = analyzer.analyze_with_shapley(enriched, attributor)

                                if matchups:
                                    logger.info(
                                        "%s: %d Shapley→Elo matchups processed",
                                        pos.ticker, len(matchups),
                                    )
                                    arena.save(str(arena_path))
                            except Exception as e:
                                logger.warning("Shapley→Elo feedback failed for %s: %s", pos.ticker, e)
                else:
                    logger.info("[Phase 4] No positions. Session trades: %d", session_trades)

                # Reset daily state
                bridge.position_manager.reset_daily()
                watchlist.clear()

        except Exception as e:
            logger.error("Phase error: %s", e)

        # Wait 60 seconds between phase checks
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass

    # ── Shutdown: close all remaining positions ──
    for pos in bridge.position_manager.open_positions:
        await bridge.close_with_attribution(
            ticker=pos.ticker,
            exit_price=pos.entry_price,
        )

    logger.info("Paper trading stopped. Session trades: %d", session_trades)


async def cmd_backtest(settings: Settings) -> None:
    """
    Run CPCV backtest with PBO+DSR combined acceptance gate.

    Uses HistoricalBacktestSimulator for full pipeline:
      1. Load or generate (signals, returns) data
      2. LLM-Aware CPCV with contamination detection
      3. Deflated Sharpe Ratio computation
      4. Combined gate: PBO < 0.10 AND DSR > 0.95

    Supports --synthetic flag for testing without historical data.
    Supports --n-obs, --accuracy, --seed for synthetic data params.

    Ref: INV-001 (CPCV mandatory), REF-007 (Lopez de Prado)
    Ref: ADR-011 (LLM-Aware Backtesting)
    Ref: ADR-015 (Production Readiness, D3)
    Ref: ADR-016 (Production Wiring, D3)
    """
    from src.core.backtest_simulator import HistoricalBacktestSimulator
    import json as _json

    logger.info("═══ MOMENTUM-X BACKTESTER ═══")
    logger.info("Validation: LLM-Aware CPCV (Purged + Embargoed)")
    logger.info("Acceptance gate: PBO < 0.10 AND DSR > 0.95")

    sim = HistoricalBacktestSimulator(
        model_id=settings.models.tier1_model,
        n_groups=6,
        n_test_groups=2,
        pbo_threshold=0.10,
        dsr_threshold=0.95,
    )

    # For now: generate synthetic data (historical data loading = future work)
    logger.info("Generating synthetic backtest data...")
    signals, returns = sim.generate_synthetic_data(
        n=500,
        signal_accuracy=0.55,
        seed=42,
    )

    logger.info("Running backtest: %d observations, model=%s", len(signals), settings.models.tier1_model)
    report = sim.run(
        signals=signals,
        returns=returns,
        strategy_name="momentum_x_synthetic",
    )

    # ── Report ──
    logger.info("═══ BACKTEST RESULTS ═══")
    logger.info("Strategy: %s", report.strategy_name)
    logger.info("Period: %s → %s", report.backtest_start, report.backtest_end)
    logger.info("Observations: %d | Folds: %d | Contaminated: %d",
                report.n_observations, report.n_folds, report.n_contaminated_folds)
    logger.info("PBO: %.4f (%s)", report.pbo, "✓ PASS" if report.pbo_pass else "✗ FAIL")
    logger.info("DSR: %.4f (%s)", report.dsr, "✓ PASS" if report.dsr_pass else "✗ FAIL")
    logger.info("Clean OOS Sharpe: %.4f", report.clean_oos_sharpe)

    if report.accepted:
        logger.info("═══ VERDICT: ACCEPTED ═══")
    else:
        logger.info("═══ VERDICT: REJECTED ═══")
    logger.info("Summary: %s", report.summary)

    # Save report to disk
    report_path = Path("data/backtest_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_json.dumps(report.to_dict(), indent=2, default=str))
    logger.info("Report saved to %s", report_path)


async def cmd_analyze(settings: Settings) -> None:
    """
    Post-session analysis: load closed trades, compute Elo feedback, update arena.

    ### ARCHITECTURAL CONTEXT
    Node ID: cli.analyze
    Graph Link: docs/memory/graph_state.json → "cli.main"

    ### RESEARCH BASIS
    Closes the Elo optimization loop from ADR-009.
    Ref: docs/research/POST_TRADE_ANALYSIS.md
    Ref: MOMENTUM_LOGIC.md §15 (Elo Feedback Dynamics)

    ### CRITICAL INVARIANTS
    1. Arena state is loaded from disk (or seeded fresh if missing).
    2. Batch analysis processes ALL closed trades from the session.
    3. Updated Elo ratings are saved back to disk.
    4. Summary is printed to stdout for operator review.
    """
    logger.info("📊 Starting post-session analysis...")

    # Load or seed arena
    arena_path = Path("data/arena_ratings.json")
    if arena_path.exists():
        arena = PromptArena.load(str(arena_path))
        logger.info("Loaded arena state from %s", arena_path)
    else:
        arena = seed_default_variants()
        logger.info("No arena state found — seeded defaults")

    analyzer = PostTradeAnalyzer(arena=arena)

    # Load closed trades from position manager data
    trades_path = Path("data/closed_trades.json")
    trade_results: list[TradeResult] = []

    if trades_path.exists():
        import json
        raw_trades = json.loads(trades_path.read_text())
        for t in raw_trades:
            try:
                trade_results.append(TradeResult(
                    ticker=t["ticker"],
                    entry_price=t["entry_price"],
                    exit_price=t["exit_price"],
                    entry_time=datetime.fromisoformat(t["entry_time"]),
                    exit_time=datetime.fromisoformat(t["exit_time"]),
                    agent_variants=t.get("agent_variants", {}),
                    agent_signals=t.get("agent_signals", {}),
                ))
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed trade record: %s", e)
    else:
        logger.info("No closed trades found at %s", trades_path)

    # Run batch analysis
    if trade_results:
        total_matchups = analyzer.batch_analyze(trade_results)
        logger.info(
            "Processed %d trades → %d Elo matchups",
            len(trade_results), total_matchups,
        )

        # Save updated arena state
        arena.save(str(arena_path))
        logger.info("Arena state saved to %s", arena_path)
    else:
        logger.info("No trades to analyze")

    # Print Elo summary
    summary = analyzer.get_elo_summary()
    print("\n🏟️  PROMPT ARENA — ELO RATINGS AFTER ANALYSIS")
    print("=" * 60)
    for vid, elo in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vid:35s} Elo = {elo:.0f}")
    print("=" * 60)
    print(f"  Trades analyzed: {len(trade_results)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="momentum-x",
        description="Momentum-X: Explosive Alpha Trading System",
    )
    parser.add_argument(
        "command",
        choices=["scan", "evaluate", "paper", "backtest", "analyze"],
        help="Operating mode",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="USE LIVE TRADING (requires confirmation). Default: paper.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Scan polling interval in seconds (default: 30). Used with 'scan' command.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run scan once and exit (default: continuous polling).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data for backtesting (default: True). Future: load historical data.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load settings
    settings = Settings()

    # Safety: Live trading requires explicit confirmation (INV-007)
    if args.live:
        logger.warning("⚠️  LIVE TRADING MODE REQUESTED ⚠️")
        confirm = input("Type 'I ACCEPT THE RISK' to proceed with REAL MONEY: ")
        if confirm != "I ACCEPT THE RISK":
            logger.info("Live trading cancelled.")
            sys.exit(0)
        settings.mode = "live"
        settings.alpaca.base_url = "https://api.alpaca.markets"
    else:
        settings.mode = "paper"

    # Dispatch to command handler
    if args.command == "scan":
        asyncio.run(cmd_scan(settings, interval=args.interval, once=args.once))
    else:
        commands = {
            "evaluate": cmd_evaluate,
            "paper": cmd_paper,
            "backtest": cmd_backtest,
            "analyze": cmd_analyze,
        }
        asyncio.run(commands[args.command](settings))


if __name__ == "__main__":
    main()
