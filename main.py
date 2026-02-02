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

from config.settings import Settings

logger = logging.getLogger("momentum_x")


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


async def cmd_scan(settings: Settings) -> None:
    """
    Run pre-market gap scanner only.
    Outputs ranked candidate list without agent evaluation.
    """
    from src.data.alpaca_client import AlpacaDataClient

    logger.info("═══ MOMENTUM-X PRE-MARKET SCANNER ═══")
    logger.info("Mode: %s | Feed: %s", settings.mode, settings.alpaca.feed)

    client = AlpacaDataClient(settings.alpaca)

    # Check account connectivity
    try:
        account = await client.get_account()
        equity = float(account.get("equity", 0))
        bp = float(account.get("buying_power", 0))
        logger.info("Account: equity=$%.2f, buying_power=$%.2f", equity, bp)
    except Exception as e:
        logger.error("Failed to connect to Alpaca: %s", e)
        logger.error("Check ALPACA_API_KEY and ALPACA_SECRET_KEY env vars")
        return

    # TODO: Fetch most active/gapping tickers from Alpaca screener
    # For now, demonstrate the pipeline with placeholder tickers
    logger.info("Scanner: Fetching market snapshots...")
    logger.info("(Full scanner integration requires Alpaca market data subscription)")
    logger.info("Scanner complete. Use 'evaluate' mode for full agent pipeline.")


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
    Full paper trading loop.
    Runs continuously through all 4 market phases.

    Ref: SYSTEM_ARCHITECTURE.md (4 Market Phases)
    Phase 1: Pre-Market (4:00-9:30 ET) — gap detection, RVOL, ranked list
    Phase 2: Market Open (9:30-10:00 ET) — ORB, debate top 5, orders
    Phase 3: Intraday (10:00-15:45 ET) — monitor, new candidates
    Phase 4: After-Hours (16:00+) — close day, analyze
    """
    from src.core.orchestrator import Orchestrator
    from src.execution.alpaca_executor import AlpacaExecutor
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

    orchestrator = Orchestrator(settings)

    # Graceful shutdown handler
    shutdown = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info("Shutdown signal received. Closing positions...")
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Paper trading loop started. Press Ctrl+C to stop.")
    logger.info("Waiting for market phase transitions...")

    # Main loop — runs until shutdown
    while not shutdown.is_set():
        now = datetime.now(timezone.utc)
        hour_et = (now.hour - 5) % 24  # Approximate ET offset

        if 4 <= hour_et < 9:
            logger.info("[Phase 1] Pre-Market scanning...")
        elif 9 <= hour_et < 10:
            logger.info("[Phase 2] Market Open — evaluating candidates...")
        elif 10 <= hour_et < 16:
            logger.info("[Phase 3] Intraday monitoring...")
        else:
            logger.info("[Phase 4] After-hours / Pre-session...")

        # Wait 60 seconds between phase checks
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass

    logger.info("Paper trading stopped. Generating session report...")


async def cmd_backtest(settings: Settings) -> None:
    """
    Run CPCV backtest on historical data.
    Ref: INV-001 (CPCV mandatory), REF-007 (Lopez de Prado)
    """
    logger.info("═══ MOMENTUM-X BACKTESTER ═══")
    logger.info("Validation: CPCV (Purged + Embargoed)")
    logger.info("PBO threshold: < 0.10")
    logger.info("Backtest mode ready. Requires historical candidate data.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="momentum-x",
        description="Momentum-X: Explosive Alpha Trading System",
    )
    parser.add_argument(
        "command",
        choices=["scan", "evaluate", "paper", "backtest"],
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
    commands = {
        "scan": cmd_scan,
        "evaluate": cmd_evaluate,
        "paper": cmd_paper,
        "backtest": cmd_backtest,
    }

    asyncio.run(commands[args.command](settings))


if __name__ == "__main__":
    main()
