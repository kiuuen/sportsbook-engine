import asyncio
import logging
import signal
from typing import Any, List, Optional, Sequence

from pipeline.orchestrator import MultiTargetOrchestrator
from state import EventSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _snapshot_provider_count(snapshot: EventSnapshot) -> int:
    """Count distinct provider domains contributing to one event snapshot."""
    return len(snapshot.providers)

def _format_sample_line(snapshots: Sequence[EventSnapshot]) -> str:
    """Render one compact sample line of converged odds from live snapshots.

    Prefers the first multi-book converged event so the monitor showcases
    cross-provider agreement; falls back to the first available event.
    """
    if not snapshots:
        return "no live events held in memory"
    sample = next(
        (s for s in snapshots if len(s.providers) > 1), snapshots[0]
    )

    teams = " vs ".join(
        team for team in (sample.home_team, sample.away_team) if team
    ) or sample.event_id[:12]
    kickoff = sample.start_time or "unknown kickoff"

    if not sample.markets:
        return f"{teams} ({kickoff}) | no markets ingested yet"

    market = sample.markets[0]
    line_part = f" {market.line}" if market.line is not None else ""
    quote_parts: List[str] = []
    for provider_domain, outcomes in sorted(market.quotes.items()):
        if not outcomes:
            continue
        outcome = outcomes[0]
        odds = (
            f"{outcome.odds_decimal:.3f}"
            if outcome.odds_decimal is not None
            else str(outcome.odds_raw)
        )
        quote_parts.append(f"{provider_domain}={outcome.label}@{odds}")

    if not quote_parts:
        return f"{teams} ({kickoff}) | market '{market.canonical_type}{line_part}' has no quotes"

    text = (
        f"{teams} ({kickoff}) | {market.canonical_type}{line_part} | "
        + " | ".join(quote_parts)
    )
    if len(text) > 220:
        text = text[:220] + "..."
    return text

def _print_status_summary(orchestrator: MultiTargetOrchestrator) -> None:
    """Print a one-line summary of the current state of the orchestrator."""
    snapshots = orchestrator.get_live_snapshots()
    total_events = len(snapshots)
    converged = sum(1 for s in snapshots if _snapshot_provider_count(s) > 1)
    sample_line = _format_sample_line(snapshots)
    print(f"[*] Events: {total_events} | Converged: {converged} | Sample: {sample_line}")

async def _monitor_loop(orchestrator: MultiTargetOrchestrator) -> None:
    """Run the status monitor loop until cancelled."""
    while True:
        _print_status_summary(orchestrator)
        await asyncio.sleep(1.0)

async def main() -> None:
    """Run the orchestrator and monitor until shutdown."""
    orchestrator = MultiTargetOrchestrator()
    monitor_task: Optional[asyncio.Task] = None

    async def _shutdown_handler() -> None:
        """Handle shutdown signals by stopping the orchestrator."""
        if monitor_task is not None:
            monitor_task.cancel()
        await orchestrator.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown_handler()))

    try:
        monitor_task = asyncio.create_task(_monitor_loop(orchestrator))
        await orchestrator.start(register_signals=False)
    except Exception as err:
        logging.error(f"Orchestrator failed: {err}")
        raise
    finally:
        if monitor_task is not None:
            monitor_task.cancel()
        await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())
