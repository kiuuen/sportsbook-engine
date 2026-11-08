"""Top-level application dispatcher for the sportsbook ingestion engine.

Routes CLI subcommands to the operational tools of the repository:

- ``run``: live multi-target ingestion with a periodic status monitor.
- ``discover``: interactive Chrome/CDP schema discovery (schema_exporter).
- ``check``: replay health-check CLI (replay_executor).
- ``norm``: normalization diagnostic harness (test_norm).
- ``targets``: list targets registered in the control-plane database.

Executed without arguments, an interactive menu offers the same actions.

Signal handling is cross-platform: ``loop.add_signal_handler`` is used where
supported (Unix); on Windows, where the asyncio API raises
``NotImplementedError``, a classic ``signal.signal`` fallback marshals the
shutdown coroutine onto the running event loop instead.
"""

import argparse
import asyncio
import logging
import os
import pathlib
import signal
import sqlite3
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from browser_orchestrator import ChromeOrchestrator
from database_manager import DatabaseManager
from pipeline.orchestrator import MultiTargetOrchestrator
from state import EventSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent

# Monitor/display and infrastructure defaults. Every value is overridable via
# CLI flags or environment variables; no magic constants at call sites.
_DEFAULT_MONITOR_INTERVAL_SECONDS: float = float(
    os.environ.get("MONITOR_INTERVAL_SECONDS", "1.0")
)
_DEFAULT_SAMPLE_MAX_WIDTH: int = int(os.environ.get("MONITOR_SAMPLE_MAX_WIDTH", "220"))
_DEFAULT_DB_PATH: str = os.environ.get("CONTROL_PLANE_DB", "./data/control_plane.db")
_DISCOVERY_CDP_PORT: int = int(os.environ.get("DISCOVERY_CDP_PORT", "9222"))

_TOOL_SCRIPTS: Dict[str, str] = {
    "discover": "schema_exporter.py",
    "check": "replay_executor.py",
    "norm": "test_norm.py",
}

_MENU_CHOICES: Dict[str, str] = {
    "1": "run",
    "2": "discover",
    "3": "check",
    "4": "norm",
    "5": "targets",
}


# ---------------------------------------------------------------------- #
# Live monitor rendering
# ---------------------------------------------------------------------- #

def _format_sample_line(
    snapshots: Sequence[EventSnapshot], max_width: int
) -> str:
    """Render one compact sample line of converged odds from live snapshots.

    Prefers the first multi-book converged event so the monitor showcases
    cross-provider agreement; falls back to the first available event.
    """
    if not snapshots:
        return "no live events held in memory"
    sample = next((s for s in snapshots if len(s.providers) > 1), snapshots[0])

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
    if len(text) > max_width:
        text = text[:max_width] + "..."
    return text


def _print_status_summary(
    orchestrator: MultiTargetOrchestrator, max_width: int
) -> None:
    """Print a one-line summary of the current state of the orchestrator.

    Aggregates granular in-memory statistics from the live snapshots:
    total events, total markets, total individual odds values held in
    memory, and the count of multi-provider converged events.
    """
    snapshots = orchestrator.get_live_snapshots()
    total_events = len(snapshots)
    total_markets = 0
    total_odds_values = 0
    converged_events = 0

    for snapshot in snapshots:
        markets = snapshot.markets
        total_markets += len(markets)
        for market in markets:
            for outcomes in market.quotes.values():
                total_odds_values += sum(
                    1 for outcome in outcomes if outcome.odds_decimal is not None
                )
        if len(snapshot.providers) > 1:
            converged_events += 1

    sample_line = _format_sample_line(snapshots, max_width=max_width)
    print(
        f"[*] Events: {total_events} | Markets: {total_markets} | "
        f"Odds values: {total_odds_values} | Converged: {converged_events} | "
        f"Sample: {sample_line}"
    )


async def _monitor_loop(
    orchestrator: MultiTargetOrchestrator,
    interval_seconds: float,
    sample_max_width: int,
) -> None:
    """Run the status monitor loop until cancelled."""
    while True:
        _print_status_summary(orchestrator, max_width=sample_max_width)
        await asyncio.sleep(interval_seconds)


# ---------------------------------------------------------------------- #
# Cross-platform lifecycle management
# ---------------------------------------------------------------------- #

def _install_signal_handlers(
    shutdown_callback: Callable[[], "asyncio.Future"],
) -> bool:
    """Install cross-platform shutdown handlers on the running event loop.

    Prefers ``loop.add_signal_handler`` (Unix). On Windows, where that API
    raises ``NotImplementedError``, falls back to classic ``signal.signal``
    handlers that marshal the shutdown coroutine onto the event loop via
    ``loop.call_soon_threadsafe``.

    Returns:
        True when at least one shutdown signal handler was installed.
    """
    loop = asyncio.get_running_loop()
    installed = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(shutdown_callback())
            )
            installed = True
            continue
        except (NotImplementedError, RuntimeError, ValueError):
            pass  # Windows Proactor loop or non-main-thread loop: fall back.

        def _fallback_handler(signum: int, frame: Any) -> None:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(shutdown_callback())
            )

        try:
            signal.signal(sig, _fallback_handler)
            installed = True
        except (OSError, ValueError, RuntimeError) as err:
            logger.debug(
                f"[main] Signal handler for {sig} unavailable: {err}"
            )
    return installed


def _cleanup_discovery_browser() -> None:
    """Terminate any orphan Chrome/CDP process on the discovery debug port.

    Reuses :meth:`ChromeOrchestrator._cleanup_port_if_busy`, which detects a
    process listening on the CDP port and force-terminates it, preventing
    orphaned browser subprocesses after dispatcher shutdown. Failures are
    logged and never propagate.
    """
    try:
        cleanup_orchestrator = ChromeOrchestrator(
            profile_name="main_dispatcher_cleanup",
            port=_DISCOVERY_CDP_PORT,
            headless=False,
        )
        cleanup_orchestrator._cleanup_port_if_busy()
        cleanup_orchestrator.stop()
    except Exception as err:
        logger.warning(f"[main] CDP port cleanup failed: {err}")


async def _run_ingestion(args: argparse.Namespace) -> None:
    """Run the orchestrator and monitor until shutdown."""
    db_manager: Optional[DatabaseManager] = None
    try:
        db_manager = DatabaseManager(db_path=args.db)
    except Exception as err:
        logger.error(f"[main] Failed to open control-plane database: {err}")
        return

    orchestrator = MultiTargetOrchestrator(db_manager=db_manager)
    monitor_task: Optional["asyncio.Task"] = None
    shutdown_started = False

    async def _shutdown() -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        if monitor_task is not None:
            monitor_task.cancel()
        await orchestrator.stop()

    _install_signal_handlers(_shutdown)

    try:
        monitor_task = asyncio.create_task(
            _monitor_loop(
                orchestrator,
                interval_seconds=args.monitor_interval,
                sample_max_width=args.sample_width,
            )
        )
        await orchestrator.start(register_signals=False)
    finally:
        if not shutdown_started:
            shutdown_started = True
            if monitor_task is not None:
                monitor_task.cancel()
            try:
                await orchestrator.stop()
            except Exception as err:
                logger.error(f"[main] Orchestrator shutdown error: {err}")
        if db_manager is not None:
            db_manager.close()
        _cleanup_discovery_browser()


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand: live ingestion with status monitor."""
    try:
        asyncio.run(_run_ingestion(args))
    except KeyboardInterrupt:
        # Reached only when no signal handler could be installed at all;
        # the finally-block inside _run_ingestion already ran cleanup.
        logger.warning("[main] Interrupted before graceful shutdown completed.")
        return 130
    return 0


# ---------------------------------------------------------------------- #
# Tool-script dispatch (discover / check / norm)
# ---------------------------------------------------------------------- #

def _run_tool_script(script_name: str, script_args: Sequence[str]) -> int:
    """Run one repository tool script in a child Python interpreter.

    The tool scripts are interactive, self-contained CLIs; dispatching them
    as subprocesses isolates their own signal handling and exit codes from
    the dispatcher process.
    """
    script_path = _PROJECT_ROOT / script_name
    if not script_path.is_file():
        logger.error(f"[main] Tool script not found: {script_path}")
        return 1
    command: List[str] = [sys.executable, str(script_path), *script_args]
    try:
        completed = subprocess.run(command, check=False)
        return int(completed.returncode)
    except KeyboardInterrupt:
        logger.info("[main] Tool interrupted by operator.")
        return 130
    except OSError as err:
        logger.error(f"[main] Failed to launch tool '{script_name}': {err}")
        return 1


def _cmd_discover(args: argparse.Namespace) -> int:
    """Execute the ``discover`` subcommand: interactive schema discovery."""
    return _run_tool_script(_TOOL_SCRIPTS["discover"], [])


def _cmd_check(args: argparse.Namespace) -> int:
    """Execute the ``check`` subcommand: replay health-check CLI."""
    return _run_tool_script(_TOOL_SCRIPTS["check"], [])


def _cmd_norm(args: argparse.Namespace) -> int:
    """Execute the ``norm`` subcommand: normalization diagnostic harness."""
    return _run_tool_script(_TOOL_SCRIPTS["norm"], args.target_args)


# ---------------------------------------------------------------------- #
# Targets listing
# ---------------------------------------------------------------------- #

def _cmd_targets(args: argparse.Namespace) -> int:
    """Execute the ``targets`` subcommand: list registered targets."""
    try:
        db = DatabaseManager(db_path=args.db)
    except Exception as err:
        logger.error(f"[main] Failed to open control-plane database: {err}")
        return 1
    try:
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT id, domain, provider_family, discovered_at "
            "FROM targets ORDER BY id ASC"
        ).fetchall()
    except sqlite3.Error as err:
        logger.error(f"[main] Failed to query registered targets: {err}")
        return 1
    finally:
        db.close()

    if not rows:
        print("[-] No targets registered in the control-plane database.")
        return 0

    print("\n=======================================================")
    print("              REGISTERED TARGETS                       ")
    print("=======================================================")
    for row in rows:
        print(
            f"    [{row['id']}] {row['domain']} | "
            f"provider={row['provider_family']} | "
            f"discovered={row['discovered_at']}"
        )
    print(f"    Total: {len(rows)} target(s)\n")
    return 0


# ---------------------------------------------------------------------- #
# CLI plumbing
# ---------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="sportsbook_engine",
        description="Sportsbook ingestion engine application dispatcher.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Start live multi-target ingestion with the status monitor."
    )
    run_parser.add_argument(
        "--monitor-interval",
        type=float,
        default=_DEFAULT_MONITOR_INTERVAL_SECONDS,
        help="Seconds between status summary prints "
        f"(default: {_DEFAULT_MONITOR_INTERVAL_SECONDS}).",
    )
    run_parser.add_argument(
        "--sample-width",
        type=int,
        default=_DEFAULT_SAMPLE_MAX_WIDTH,
        help="Maximum character width of the sample odds line "
        f"(default: {_DEFAULT_SAMPLE_MAX_WIDTH}).",
    )
    run_parser.add_argument(
        "--db",
        default=_DEFAULT_DB_PATH,
        help=f"Control-plane SQLite database path (default: {_DEFAULT_DB_PATH}).",
    )

    subparsers.add_parser(
        "discover", help="Launch interactive Chrome/CDP schema discovery."
    )
    subparsers.add_parser(
        "check", help="Run the interactive replay health-check CLI."
    )
    norm_parser = subparsers.add_parser(
        "norm", help="Run the normalization diagnostic harness."
    )
    norm_parser.add_argument(
        "target_args",
        nargs="*",
        help="Arguments forwarded to test_norm.py: <target_domain> [endpoint_filter].",
    )
    targets_parser = subparsers.add_parser(
        "targets", help="List targets registered in the control-plane database."
    )
    targets_parser.add_argument(
        "--db",
        default=_DEFAULT_DB_PATH,
        help=f"Control-plane SQLite database path (default: {_DEFAULT_DB_PATH}).",
    )
    return parser


def _interactive_menu() -> Optional[str]:
    """Prompt the operator to choose an action; None when nothing selected."""
    print("\n=======================================================")
    print("           SPORTSBOOK ENGINE DISPATCHER                ")
    print("=======================================================")
    print("    [1] run      - Start live multi-target ingestion")
    print("    [2] discover - Interactive Chrome schema discovery")
    print("    [3] check    - Replay health checks")
    print("    [4] norm     - Normalization diagnostic test")
    print("    [5] targets  - List registered targets")
    try:
        choice = input("\nSelect an action: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    selected = _MENU_CHOICES.get(choice)
    if selected is None:
        print(f"[-] Invalid selection: '{choice}'.")
    return selected


_COMMAND_HANDLERS: Dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _cmd_run,
    "discover": _cmd_discover,
    "check": _cmd_check,
    "norm": _cmd_norm,
    "targets": _cmd_targets,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch the requested subcommand; returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        command = _interactive_menu()
        if command is None:
            print("[*] No action selected; exiting.")
            return 0
        # Re-parse so the chosen subcommand receives its own defaults.
        args = parser.parse_args([command])

    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
