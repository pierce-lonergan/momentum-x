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
<<<<<<< HEAD
    """
    try:
        # Use Alpaca most-active or screener endpoint if available
        # For now, fetch snapshots for a watchlist or most-active tickers
=======

    Uses AlpacaDataClient.get_snapshots() which returns normalized data:
    {ticker: {last_price, prev_close, volume, bid, ask, ...}}

    Ref: DATA-001 (Alpaca snapshot endpoint)
    Ref: ADR-002 (Snapshot-based architecture)
    """
    try:
        # Get most-active tickers for scanning universe
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
        tickers = await client.get_most_active_tickers(limit=50)
        if not tickers:
            return {}

<<<<<<< HEAD
=======
        # Batch snapshot fetch via normalized client
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
        snapshots = await client.get_snapshots(tickers)
        quotes = {}
        for ticker, snap in snapshots.items():
            try:
<<<<<<< HEAD
                current = float(snap.get("latestTrade", {}).get("p", 0) or
                               snap.get("minuteBar", {}).get("c", 0))
                prev_close = float(snap.get("prevDailyBar", {}).get("c", 0))
                volume = int(snap.get("minuteBar", {}).get("v", 0) or
                            snap.get("dailyBar", {}).get("v", 0))
                avg_vol = max(1, int(snap.get("prevDailyBar", {}).get("v", 1)))
=======
                # Use normalized field names from AlpacaDataClient._normalize_snapshot
                current = float(snap.get("last_price", 0))
                prev_close = float(snap.get("prev_close", 0))
                volume = int(snap.get("volume", 0))
                avg_vol = max(1, int(snap.get("prev_volume", 1)))
                bid = float(snap.get("bid", 0))
                ask = float(snap.get("ask", 0))

                # Fallback to raw Alpaca fields if normalized fields missing
                if current <= 0:
                    current = float(
                        snap.get("latestTrade", {}).get("p", 0) or
                        snap.get("minuteBar", {}).get("c", 0)
                    )
                if prev_close <= 0:
                    prev_close = float(snap.get("prevDailyBar", {}).get("c", 0))
                if volume <= 0:
                    volume = int(
                        snap.get("minuteBar", {}).get("v", 0) or
                        snap.get("dailyBar", {}).get("v", 0)
                    )
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)

                if current > 0 and prev_close > 0:
                    quotes[ticker] = {
                        "current_price": current,
                        "previous_close": prev_close,
                        "premarket_volume": volume,
                        "avg_volume_at_time": avg_vol,
<<<<<<< HEAD
                        "float_shares": None,
                        "market_cap": None,
=======
                        "float_shares": snap.get("float_shares"),
                        "market_cap": snap.get("market_cap"),
                        "bid": bid,
                        "ask": ask,
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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
<<<<<<< HEAD
=======
    from src.execution.tranche_monitor import TrancheExitMonitor, TrancheFillEvent
    from src.execution.stop_resubmitter import StopResubmitter
    from src.execution.fill_stream_bridge import FillStreamBridge
    from src.execution.portfolio_risk import PortfolioRiskManager
    from src.monitoring.server import MetricsServer
    from src.monitoring.metrics import reset_metrics
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)

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
<<<<<<< HEAD
=======
    tranche_monitor = TrancheExitMonitor(position_manager=position_manager)
    stop_resubmitter = StopResubmitter(client=client)
    fill_bridge = FillStreamBridge(
        tranche_monitor=tranche_monitor,
        stop_resubmitter=stop_resubmitter,
    )
    portfolio_risk = PortfolioRiskManager(
        max_sector_positions=2,
        max_portfolio_heat_pct=5.0,
    )

    # ── Launch WebSocket fill stream as background task (ADR-023, D2) ──
    from src.data.websocket_client import TradeUpdatesStream
    trade_stream = TradeUpdatesStream(
        api_key=settings.alpaca.api_key,
        secret_key=settings.alpaca.secret_key,
        paper=(settings.mode == "paper"),
    )
    trade_stream.on_trade_update = fill_bridge.on_trade_update
    fill_stream_task: asyncio.Task | None = None
    try:
        fill_stream_task = asyncio.create_task(trade_stream.connect())
        logger.info("WebSocket fill stream launched as background task")
    except Exception as e:
        logger.warning("WebSocket fill stream failed to launch: %s — falling back to polling", e)
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)

    # Track candidates and their ScoredCandidates for execution
    watchlist: list = []  # CandidateStock from Phase 1
    session_trades: int = 0

<<<<<<< HEAD
=======
    # ── Start Metrics HTTP Server (ADR-019) ──
    reset_metrics()
    metrics_server = MetricsServer(port=9090)
    metrics_server.start()
    logger.info("Metrics server: http://0.0.0.0:9090/metrics")

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
    # Graceful shutdown handler
    shutdown = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info("Shutdown signal received. Closing positions...")
<<<<<<< HEAD
=======
        metrics_server.stop()
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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
<<<<<<< HEAD
=======
                            # ── Portfolio risk check (ADR-024) ──
                            risk_check = portfolio_risk.check_entry(
                                ticker=verdict.ticker,
                                stop_loss_pct=2.0,  # Default stop distance
                                positions=bridge.position_manager.open_positions,
                            )
                            if not risk_check.allowed:
                                logger.warning(
                                    "PORTFOLIO RISK BLOCKED: %s — %s",
                                    verdict.ticker, risk_check.reason,
                                )
                                metrics.risk_vetoes.inc()
                                continue

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
                            # Retrieve scored from orchestrator (cached during eval)
                            order = await bridge.execute_verdict(verdict, scored=None)
                            if order is not None:
                                session_trades += 1
                                logger.info(
                                    "ORDER FILLED: %s qty=%d @ $%.2f (order_id=%s)",
                                    order.ticker, order.qty, order.submitted_price, order.order_id,
                                )

<<<<<<< HEAD
            elif 10 <= hour_et < 16:
                # ── Phase 3: Intraday Monitoring ──
=======
                                # ── Submit tranche exit orders (ADR-020, D2) ──
                                try:
                                    position = None
                                    for p in bridge.position_manager.open_positions:
                                        if p.ticker == order.ticker:
                                            position = p
                                            break

                                    if position is not None:
                                        tranches = bridge.position_manager.compute_exit_tranches(position)
                                        for tranche in tranches:
                                            tranche_resp = await client.submit_limit_order(
                                                symbol=order.ticker,
                                                qty=tranche.qty,
                                                side="sell",
                                                limit_price=tranche.target,
                                            )
                                            tranche_oid = tranche_resp.get("id", "")
                                            tranche_monitor.register_tranche_order(
                                                order_id=tranche_oid,
                                                ticker=order.ticker,
                                                tranche_number=tranche.tranche_number,
                                                target_price=tranche.target,
                                                qty=tranche.qty,
                                            )
                                            logger.info(
                                                "TRANCHE T%d submitted: %s sell %d @ $%.2f (oid=%s)",
                                                tranche.tranche_number, order.ticker,
                                                tranche.qty, tranche.target, tranche_oid,
                                            )
                                except Exception as e:
                                    logger.warning("Tranche order submission error: %s", e)

                                # ── Register initial stop for ratcheting (ADR-022, D2) ──
                                try:
                                    stop_resubmitter.register_stop(
                                        ticker=order.ticker,
                                        order_id=order.order_id,  # Bracket stop leg ID
                                        stop_price=verdict.stop_loss,
                                        qty=order.qty,
                                    )
                                    logger.info(
                                        "STOP registered: %s @ $%.2f (oid=%s)",
                                        order.ticker, verdict.stop_loss, order.order_id,
                                    )
                                except Exception as e:
                                    logger.warning("Stop registration error: %s", e)

            elif 10 <= hour_et < 16:
                # ── Phase 3: Intraday Monitoring + VWAP Breakout Scanner ──
                from src.scanners.intraday_vwap import IntradayVWAPScanner

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
                positions = bridge.position_manager.open_positions
                logger.info("[Phase 3] Intraday — monitoring %d positions", len(positions))
                for pos in positions:
                    logger.info(
                        "  %s: qty=%d, entry=$%.2f, stop=$%.2f",
                        pos.ticker, pos.remaining_qty, pos.entry_price, pos.stop_loss,
                    )

<<<<<<< HEAD
=======
                # ── Process tranche fills via WebSocket bridge (ADR-023, D2) ──
                if tranche_monitor.registered_orders > 0:
                    try:
                        # Primary path: drain WebSocket fill events (sub-second)
                        fill_events = await fill_bridge.drain_and_resubmit()
                        for fev in fill_events:
                            logger.info(
                                "TRANCHE T%d FILL (stream): %s @ $%.2f | Stop: $%.2f → $%.2f%s",
                                fev.tranche_number or 0, fev.ticker,
                                fev.filled_price, fev.old_stop or 0, fev.new_stop or 0,
                                " [RATCHETED]" if fev.stop_resubmitted else "",
                            )

                        # Fallback: position-polling if no WebSocket events detected
                        if not fill_events:
                            alpaca_positions = await client.get_positions()
                            for pos in bridge.position_manager.open_positions:
                                for ap in alpaca_positions:
                                    if ap.get("symbol") == pos.ticker:
                                        live_qty = int(ap.get("qty", 0))
                                        if live_qty < pos.remaining_qty:
                                            filled_qty = pos.remaining_qty - live_qty
                                            live_price = float(ap.get("current_price", pos.entry_price))
                                            for oid, to in list(tranche_monitor._order_map.items()):
                                                if to.ticker == pos.ticker and to.qty <= filled_qty:
                                                    result = tranche_monitor.on_fill(TrancheFillEvent(
                                                        order_id=oid,
                                                        ticker=pos.ticker,
                                                        filled_price=live_price,
                                                        filled_qty=to.qty,
                                                    ))
                                                    if result:
                                                        logger.info(
                                                            "TRANCHE T%d FILL (poll): %s @ $%.2f | Stop: $%.2f → $%.2f",
                                                            result.tranche_number, pos.ticker,
                                                            live_price, result.old_stop, result.new_stop,
                                                        )
                                                        if result.new_stop > result.old_stop:
                                                            try:
                                                                resubmit_result = await stop_resubmitter.resubmit(
                                                                    ticker=pos.ticker,
                                                                    new_stop_price=result.new_stop,
                                                                    new_qty=pos.remaining_qty - to.qty,
                                                                )
                                                                if resubmit_result.success:
                                                                    logger.info(
                                                                        "STOP RATCHETED (poll): %s $%.2f → $%.2f",
                                                                        pos.ticker, result.old_stop, result.new_stop,
                                                                    )
                                                            except Exception as e:
                                                                logger.warning("Stop resubmit error: %s", e)
                                                    break
                    except Exception as e:
                        logger.debug("Tranche fill check error: %s", e)

                # ── VWAP Breakout Scanner (ADR-018, D2) ──
                if bridge.position_manager.can_enter_new_position():
                    try:
                        quotes = await _fetch_scan_quotes(client, settings)
                        if quotes:
                            # Build VWAP snapshots from available data
                            vwap_snapshots: dict = {}
                            for ticker, q in quotes.items():
                                price = q.get("current_price", 0)
                                prev = q.get("previous_close", 0)
                                vol = q.get("premarket_volume", 0)
                                if price > 0 and prev > 0:
                                    vwap_snapshots[ticker] = {
                                        "price": price,
                                        "vwap": prev,  # Use prev_close as VWAP proxy until WebSocket
                                        "volume": vol,
                                        "avg_volume": max(1, q.get("avg_volume_at_time", vol)),
                                    }

                            if not hasattr(cmd_paper, "_vwap_scanner"):
                                cmd_paper._vwap_scanner = IntradayVWAPScanner()  # type: ignore[attr-defined]

                            vwap_signals = cmd_paper._vwap_scanner.scan(vwap_snapshots)  # type: ignore[attr-defined]

                            if vwap_signals:
                                logger.info(
                                    "[Phase 3] %d VWAP breakout signals detected",
                                    len(vwap_signals),
                                )
                                # Convert VWAP signals to CandidateStocks for evaluation
                                from src.core.models import CandidateStock

                                vwap_candidates = []
                                for sig in vwap_signals[:3]:  # Cap at 3
                                    vwap_candidates.append(CandidateStock(
                                        ticker=sig.ticker,
                                        current_price=sig.current_price,
                                        previous_close=sig.vwap,
                                        gap_pct=sig.breakout_pct,
                                        gap_classification="VWAP_BREAKOUT",
                                        rvol=sig.rvol_at_breakout,
                                        premarket_volume=sig.total_volume,
                                        scan_phase="intraday",
                                    ))

                                verdicts = await orchestrator.evaluate_candidates(vwap_candidates)
                                for verdict in verdicts:
                                    if verdict.action == "BUY":
                                        order = await bridge.execute_verdict(verdict, scored=None)
                                        if order is not None:
                                            session_trades += 1
                                            logger.info(
                                                "VWAP BREAKOUT ORDER: %s qty=%d @ $%.2f",
                                                order.ticker, order.qty, order.submitted_price,
                                            )
                    except Exception as e:
                        logger.warning("[Phase 3] VWAP scan error: %s", e)

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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

<<<<<<< HEAD
=======
                # ── Generate End-of-Day Session Report (ADR-022) ──
                try:
                    from src.analysis.session_report import SessionReportGenerator
                    report_gen = SessionReportGenerator(mode=settings.mode)
                    report = report_gen.generate()
                    report_gen.save(report)
                    logger.info("\n%s", report.summary_text())
                except Exception as e:
                    logger.warning("Session report generation failed: %s", e)

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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

<<<<<<< HEAD
=======
    # ── Shutdown WebSocket fill stream ──
    trade_stream.stop()
    if fill_stream_task is not None:
        fill_stream_task.cancel()
        try:
            await fill_stream_task
        except (asyncio.CancelledError, Exception):
            pass
    logger.info("WebSocket fill stream stopped")

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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
<<<<<<< HEAD
=======
    from src.data.historical_loader import HistoricalDataLoader
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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

<<<<<<< HEAD
    # For now: generate synthetic data (historical data loading = future work)
    logger.info("Generating synthetic backtest data...")
    signals, returns = sim.generate_synthetic_data(
        n=500,
        signal_accuracy=0.55,
        seed=42,
    )
=======
    # ── Data Loading: Multi-ticker, Single-ticker Historical, or Synthetic ──
    use_multi = hasattr(settings, 'backtest_tickers') and settings.backtest_tickers
    use_historical = hasattr(settings, 'backtest_ticker') and settings.backtest_ticker
    strategy_name = "momentum_x_synthetic"

    if use_multi:
        # Multi-ticker portfolio backtest
        tickers = settings.backtest_tickers
        logger.info("Loading multi-ticker historical data: %s...", tickers)
        try:
            from src.data.alpaca_client import AlpacaDataClient
            from src.data.multi_ticker_backtest import MultiTickerBacktest
            client = AlpacaDataClient(settings.alpaca)
            loader = HistoricalDataLoader(client=client)
            multi = MultiTickerBacktest(loader=loader)
            result = await multi.load_and_merge(
                tickers=tickers,
                days=settings.backtest_days,
            )
            signals, returns = result.merged_dataset.signals, result.merged_dataset.returns
            strategy_name = f"momentum_x_portfolio_{'_'.join(tickers[:3])}"
            logger.info(
                "Multi-ticker loaded: %d tickers, %d observations, %d BUY signals, %d failed",
                len(result.per_ticker), result.total_observations,
                result.total_buy_signals, len(result.failed_tickers),
            )
            if result.failed_tickers:
                logger.warning("Failed tickers: %s", result.failed_tickers)
        except Exception as e:
            logger.warning("Multi-ticker load failed (%s), falling back to synthetic", e)
            signals, returns = sim.generate_synthetic_data(n=500, signal_accuracy=0.55, seed=42)

    elif use_historical:
        # Historical mode: fetch real OHLCV from Alpaca
        ticker = settings.backtest_ticker
        logger.info("Loading historical data for %s...", ticker)
        try:
            from src.data.alpaca_client import AlpacaDataClient
            client = AlpacaDataClient(settings.alpaca)
            loader = HistoricalDataLoader(client=client)
            dataset = await loader.load(ticker=ticker, days=settings.backtest_days)
            signals, returns = dataset.signals, dataset.returns
            strategy_name = f"momentum_x_{ticker}"
            logger.info(
                "Historical data loaded: %d observations, %d BUY signals, %s → %s",
                dataset.n_observations, dataset.n_buy_signals,
                dataset.start_date, dataset.end_date,
            )
        except Exception as e:
            logger.warning("Historical data load failed (%s), falling back to synthetic", e)
            signals, returns = sim.generate_synthetic_data(n=500, signal_accuracy=0.55, seed=42)
    else:
        # Synthetic mode (default)
        logger.info("Generating synthetic backtest data...")
        signals, returns = sim.generate_synthetic_data(
            n=500,
            signal_accuracy=0.55,
            seed=42,
        )
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)

    logger.info("Running backtest: %d observations, model=%s", len(signals), settings.models.tier1_model)
    report = sim.run(
        signals=signals,
        returns=returns,
<<<<<<< HEAD
        strategy_name="momentum_x_synthetic",
=======
        strategy_name=strategy_name,
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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
<<<<<<< HEAD
=======
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker symbol for historical backtest (e.g., AAPL). Overrides --synthetic.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=252,
        help="Number of trading days for historical backtest (default: 252).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers for multi-ticker backtest (e.g., AAPL,MSFT,TSLA).",
    )
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)

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

<<<<<<< HEAD
=======
    # Backtest configuration from CLI flags
    if hasattr(args, 'ticker') and args.ticker:
        settings.backtest_ticker = args.ticker
    if hasattr(args, 'days') and args.days:
        settings.backtest_days = args.days
    if hasattr(args, 'tickers') and args.tickers:
        settings.backtest_tickers = [t.strip() for t in args.tickers.split(",")]

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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
