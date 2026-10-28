"""Terminal user interface for the high-scale data ingestion control plane.

This module implements an interactive, non-blocking control panel built on top
of ``rich`` and ``asyncio``. It provides:

- High-throughput live logging through a bounded in-memory ring buffer.
- Scalable observation of target state (session health, pipeline state and
  the latest probe outcome) loaded directly from the SQLite control plane.
- Interactive control actions: target discovery, endpoint testing, session
  refresh and pipeline start/stop.
- A fully asynchronous lifecycle so background work never blocks rendering or
  keyboard input handling.
"""

import asyncio
import datetime
import logging
import sys
import threading
import time
from collections import deque
from typing import Any, Awaitable, Callable, Coroutine, Deque, Dict, List, Optional, Tuple

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from browser_orchestrator import ChromeOrchestrator
from cdp_interceptor import CDPNetworkInterceptor, NetworkEvent
from database_manager import DatabaseManager
from heuristic_classifier import (
    ClassificationResult,
    HeuristicClassifier,
    RouteTemplateDeducer,
)
from pipeline_engine import PipelineEngine
from replay_executor import ReplayExecutor
from schema_exporter import SchemaExporter
from session_manager import SessionManager

if sys.platform.startswith("win"):
    import msvcrt
else:
    import select
    import termios
    import tty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class KeyListener:
    """Cross-platform, non-blocking single-keypress listener.

    On POSIX systems the terminal is switched to cbreak mode so that single key
    presses are delivered immediately without line buffering. On Windows the
    ``msvcrt`` console API is used instead. Key presses are read on a daemon
    thread and forwarded to the event loop through an ``asyncio.Queue`` so
    that neither the render loop nor background tasks are ever blocked.
    """

    POSIX_ARROW_SUFFIXES: Dict[str, str] = {
        "A": "UP",
        "B": "DOWN",
        "C": "RIGHT",
        "D": "LEFT",
    }
    WINDOWS_ARROW_CODES: Dict[str, str] = {
        "H": "UP",
        "P": "DOWN",
        "K": "LEFT",
        "M": "RIGHT",
    }

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_windows = sys.platform.startswith("win")
        self._termios_fd: Optional[int] = None
        self._saved_termios: Optional[Any] = None

    def start(self) -> None:
        """Start reading key presses on a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        if not self._is_windows and sys.stdin.isatty():
            self._termios_fd = sys.stdin.fileno()
            self._saved_termios = termios.tcgetattr(self._termios_fd)
            tty.setcbreak(self._termios_fd)
        self._thread = threading.Thread(
            target=self._read_loop, name="tui-key-listener", daemon=True
        )
        self._thread.start()

    def _read_loop(self) -> None:
        """Forward key presses to the event loop until stopped."""
        while not self._stop_event.is_set():
            try:
                key = self._read_single_key()
            except Exception:
                # Terminal input is no longer available; request a clean exit.
                self._submit("q")
                break
            if key is None:
                if self._is_windows:
                    time.sleep(0.05)
                continue
            if key == "":
                # EOF on stdin; request a clean exit.
                self._submit("q")
                break
            self._submit(key)

    def _read_single_key(self) -> Optional[str]:
        """Read one key press, or ``None`` when no key is available yet."""
        if self._is_windows:
            if not msvcrt.kbhit():
                return None
            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):
                suffix = msvcrt.getwch()
                return self.WINDOWS_ARROW_CODES.get(suffix, f"SPECIAL_{suffix}")
            return key

        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not ready:
            return None
        key = sys.stdin.read(1)
        if key == "":
            return ""
        if key == "\x1b":
            follow_up = select.select([sys.stdin], [], [], 0.0)[0]
            if not follow_up:
                return "ESC"
            bracket = sys.stdin.read(1)
            if bracket == "[":
                suffix = sys.stdin.read(1)
                return self.POSIX_ARROW_SUFFIXES.get(suffix, f"SPECIAL_{suffix}")
            return "ESC"
        return key

    def _submit(self, key: str) -> None:
        """Forward a key press to the event loop queue."""
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, key)
        except RuntimeError:
            # The event loop is already closed during interpreter shutdown.
            pass

    async def get(self) -> str:
        """Wait for and return the next key press."""
        return await self._queue.get()

    def stop(self) -> None:
        """Stop the reader thread and restore the original terminal mode."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._termios_fd is not None and self._saved_termios is not None:
            try:
                termios.tcsetattr(
                    self._termios_fd, termios.TCSADRAIN, self._saved_termios
                )
            except Exception:
                pass
            self._termios_fd = None
            self._saved_termios = None


class LogBuffer:
    """Fixed-size ring buffer that stores the most recent log messages."""

    def __init__(self, max_size: int = 1000) -> None:
        self.buffer: Deque[Tuple[str, str]] = deque(maxlen=max_size)

    def add(self, level: str, message: str) -> None:
        """Append a log message to the buffer."""
        self.buffer.append((level, message))

    def get_recent(self, count: int = 20) -> List[Tuple[str, str]]:
        """Return the most recent log messages, oldest first."""
        if count <= 0:
            return []
        return list(self.buffer)[-count:]


class TUITargetManager:
    """Manage target state, control actions and pipeline lifecycles for the TUI."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.active_targets: Dict[str, Dict[str, Any]] = {}
        self._pipeline_engines: Dict[str, PipelineEngine] = {}
        self._pipeline_tasks: Dict[str, asyncio.Task] = {}
        self.reload_targets()

    def reload_targets(self) -> None:
        """Reload all targets from the database while preserving live state."""
        previous = self.active_targets
        conn = self.db_manager.get_connection()
        rows = conn.execute("SELECT * FROM targets ORDER BY domain ASC").fetchall()
        reloaded: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            domain = row["domain"]
            prior_state = previous.get(domain, {})
            reloaded[domain] = {
                "id": int(row["id"]),
                "provider_family": row["provider_family"],
                "discovered_at": row["discovered_at"],
                "updated_at": row["updated_at"],
                "session_health": prior_state.get("session_health", "UNKNOWN"),
                "last_probe": prior_state.get("last_probe"),
                "pipeline_active": prior_state.get("pipeline_active", False),
            }
        self.active_targets = reloaded

    async def refresh_target_status(self, domain: str) -> None:
        """Refresh session health and the latest probe outcome for a target."""
        if domain not in self.active_targets:
            return

        target_id = int(self.active_targets[domain]["id"])

        session = self.db_manager.get_session_by_target_id(target_id)
        self.active_targets[domain]["session_health"] = (
            "ACTIVE" if session else "INACTIVE"
        )

        logs = self.db_manager.get_recent_replay_logs(target_id, limit=1)
        if logs:
            last_log = logs[0]
            self.active_targets[domain]["last_probe"] = {
                "status": last_log["status"],
                "status_code": last_log["status_code"],
                "message": last_log["message"],
                "executed_at": last_log["executed_at"],
            }

    async def discover_new_target(
        self,
        url: str,
        prompt_cb: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> bool:
        """Discover a new target by capturing live traffic and classifying endpoints.

        Args:
            url: URL of the site to discover. A scheme is added automatically
                when missing.
            prompt_cb: Optional async callback function to prompt the operator and
                pause the display.

        Returns:
            True when a schema was discovered and exported, False otherwise.
        """
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url

        orchestrator: Optional[ChromeOrchestrator] = None
        interceptor: Optional[CDPNetworkInterceptor] = None
        try:
            classifier = HeuristicClassifier()
            orchestrator = ChromeOrchestrator(
                profile_name="tui_discovery_profile",
                port=9222,
                headless=False,
            )
            await asyncio.to_thread(
                orchestrator.start, timeout=15.0, initial_url=url
            )

            ws_url = await asyncio.to_thread(
                orchestrator.get_cdp_url, target_url=url
            )
            if not ws_url:
                logger.warning(f"Failed to obtain CDP WebSocket URL for target: {url}")
                return False

            interceptor = CDPNetworkInterceptor(cdp_ws_url=ws_url)
            await interceptor.start()

            prompt_msg = (
                "Interact with the target in Chrome, then press Enter to analyze traffic..."
            )
            if prompt_cb is not None:
                await prompt_cb(prompt_msg)
            else:
                await asyncio.to_thread(input, prompt_msg + "\n")

            events = interceptor.get_captured_events()

            template_groups: Dict[str, List[NetworkEvent]] = {}
            for ev in events:
                if ev.response_body:
                    tmpl = RouteTemplateDeducer.deduce_template(ev.url)
                    template_groups.setdefault(tmpl, []).append(ev)

            results: List[ClassificationResult] = []
            for tmpl, group_events in template_groups.items():
                call_count = len(group_events)
                timestamps = [e.timestamp for e in group_events]
                time_span = (
                    max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
                )
                sample_event = group_events[0]
                res = classifier.score_payload(
                    url=sample_event.url,
                    payload_str=sample_event.response_body or "",
                    call_count=call_count,
                    time_span_seconds=time_span,
                    sample_event=sample_event,
                )
                if res.is_viable or res.is_grey_zone:
                    results.append(res)

            if not results:
                logger.warning(f"No viable endpoints found for target: {url}")
                return False

            exporter = SchemaExporter()
            saved_path, target_id = exporter.export_schema(
                target_url=url,
                classified_results=results,
                events=events,
                db_manager=self.db_manager,
            )
            if saved_path is None:
                logger.warning(
                    f"Schema export produced no output path for target: {url}"
                )
                return False

            logger.info(
                f"Successfully discovered and exported schema for: {url} (target_id: {target_id})"
            )
            self.reload_targets()
            return True

        except Exception as err:
            logger.error(f"Error while discovering target {url}: {err}")
            return False
        finally:
            if interceptor is not None:
                try:
                    await interceptor.stop()
                except Exception as stop_err:
                    logger.warning(
                        f"Error while stopping network interceptor: {stop_err}"
                    )
            if orchestrator is not None:
                try:
                    await asyncio.to_thread(orchestrator.stop)
                except Exception as stop_err:
                    logger.warning(
                        f"Error while stopping Chrome orchestrator: {stop_err}"
                    )

    async def test_endpoint(self, domain: str, endpoint_key: str) -> Tuple[bool, str]:
        """Execute a replay probe against a specific endpoint of a target."""
        if domain not in self.active_targets:
            return False, f"Target domain '{domain}' not found."

        try:
            executor = ReplayExecutor(schema_source=domain, db_manager=self.db_manager)
        except Exception as err:
            return False, f"Failed to load replay executor: {err}"

        try:
            response = executor.execute(endpoint_key)
            preview = ReplayExecutor.clamp_preview(response, max_lines=5, max_chars=500)
            return True, f"Endpoint test succeeded:\n{preview}"
        except Exception as err:
            return False, f"Endpoint test failed: {err}"

    async def refresh_session(self, domain: str) -> Tuple[bool, str]:
        """Refresh the stored session for a target."""
        if domain not in self.active_targets:
            return False, f"Target domain '{domain}' not found."

        session_manager = SessionManager(db_manager=self.db_manager)
        try:
            success, message = await session_manager.refresh_session(
                target_domain=domain,
                initial_url=f"https://{domain}",
            )
        except Exception as err:
            return False, f"Session refresh error: {err}"

        if success:
            await self.refresh_target_status(domain)
        return success, message

    async def toggle_pipeline(self, domain: str) -> Tuple[bool, str]:
        """Start or stop the ingestion pipeline for a target.

        Args:
            domain: The target domain whose pipeline should be toggled.

        Returns:
            A tuple of (success, message) describing the outcome.
        """
        if domain not in self.active_targets:
            return False, f"Target domain '{domain}' not found."

        target_state = self.active_targets[domain]

        if target_state.get("pipeline_active"):
            engine = self._pipeline_engines.pop(domain, None)
            if engine is not None:
                try:
                    await engine.stop()
                except Exception as err:
                    logger.warning(
                        f"Error while stopping pipeline for {domain}: {err}"
                    )

            task = self._pipeline_tasks.get(domain)
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
                except asyncio.TimeoutError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                except asyncio.CancelledError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise
                except Exception as err:
                    logger.warning(
                        f"Pipeline task for {domain} ended with an error: {err}"
                    )

            target_state["pipeline_active"] = False
            return True, f"Pipeline stopped for {domain}."

        engine = PipelineEngine(target_domain=domain, db_manager=self.db_manager)
        self._pipeline_engines[domain] = engine
        task = asyncio.create_task(engine.start(), name=f"pipeline-{domain}")
        self._pipeline_tasks[domain] = task

        def _on_pipeline_done(
            finished: asyncio.Task, bound_domain: str = domain
        ) -> None:
            self._handle_pipeline_done(bound_domain, finished)

        task.add_done_callback(_on_pipeline_done)
        target_state["pipeline_active"] = True
        return True, f"Pipeline started for {domain}."

    def _handle_pipeline_done(self, domain: str, task: asyncio.Task) -> None:
        """Clean up pipeline state after its background task finishes."""
        try:
            task.result()
            logger.info(f"Pipeline for target {domain} has finished.")
        except asyncio.CancelledError:
            pass
        except Exception as err:
            logger.error(f"Pipeline for target {domain} failed: {err}")
        finally:
            self._pipeline_tasks.pop(domain, None)
            self._pipeline_engines.pop(domain, None)
            if domain in self.active_targets:
                self.active_targets[domain]["pipeline_active"] = False

    async def stop_all_pipelines(self) -> None:
        """Gracefully stop every running pipeline engine."""
        for domain, engine in list(self._pipeline_engines.items()):
            try:
                await engine.stop()
            except Exception as err:
                logger.warning(f"Error while stopping pipeline for {domain}: {err}")
        self._pipeline_engines.clear()

        pending = [t for t in self._pipeline_tasks.values() if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._pipeline_tasks.clear()

        for state in self.active_targets.values():
            state["pipeline_active"] = False


class TUITaskManager:
    """Manage background tasks for the TUI."""

    def __init__(self) -> None:
        self.tasks: List[asyncio.Task] = []

    def add_task(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule a coroutine as a tracked background task."""
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        task.add_done_callback(self._handle_task_done)

    def _handle_task_done(self, task: asyncio.Task) -> None:
        """Log unexpected background task failures."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as err:
            logger.error(f"Background task failed: {err}")

    async def shutdown(self) -> None:
        """Cancel every tracked background task and wait for completion."""
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()


class TUITerminal:
    """Terminal user interface for high-scale data ingestion control."""

    def __init__(self) -> None:
        self.console = Console()
        self.layout = Layout()
        self.log_buffer = LogBuffer()
        self.db_manager = DatabaseManager()
        self.target_manager = TUITargetManager(self.db_manager)
        self.task_manager = TUITaskManager()
        self.key_listener: Optional[KeyListener] = None
        self.active = True
        self.show_help = False
        self.selected_domain: Optional[str] = None
        self._target_index: int = 0

        self._configure_layout()
        self._setup_logging_handler()

    def _configure_layout(self) -> None:
        """Configure the terminal layout regions."""
        self.layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="targets", ratio=1),
            Layout(name="logs", ratio=1),
        )
        self.layout["targets"].split_column(
            Layout(name="target_summary", size=8),
            Layout(name="target_details", ratio=1),
        )

    def _setup_logging_handler(self) -> None:
        """Route log records into the on-screen buffer and the persistent log file.

        The on-screen buffer receives short single-line messages without
        tracebacks, while the file handler records complete messages and
        exception tracebacks for later inspection.
        """

        class LogBufferHandler(logging.Handler):
            def __init__(self, buffer: LogBuffer) -> None:
                super().__init__()
                self._buffer = buffer

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    timestamp = datetime.datetime.fromtimestamp(record.created).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    self._buffer.add(record.levelname, f"{timestamp} {record.getMessage()}")
                except Exception:
                    self.handleError(record)

        root_logger = logging.getLogger()
        for existing_handler in list(root_logger.handlers):
            root_logger.removeHandler(existing_handler)

        handler = LogBufferHandler(self.log_buffer)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        self._file_log_handler = self._create_file_log_handler()
        if self._file_log_handler is not None:
            root_logger.addHandler(self._file_log_handler)

    def _render_header(self) -> RenderableType:
        """Render the header section."""
        now_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = Text("High-Scale Data Ingestion Control Panel", style="bold magenta")
        subtitle = Text(
            f"Targets tracked: {len(self.target_manager.active_targets)}"
            f"   |   {now_text}   |   Press 'h' for help",
            style="italic",
        )
        return Panel(
            Group(title, subtitle), box=box.ROUNDED, border_style="bright_blue"
        )

    def _render_target_summary(self) -> RenderableType:
        """Render the target summary table."""
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("", width=3, no_wrap=True)
        table.add_column("Domain", style="cyan", no_wrap=True)
        table.add_column("Provider", style="green")
        table.add_column("Session", style="yellow")
        table.add_column("Pipeline", style="blue")
        table.add_column("Last Probe", style="magenta")

        domains = sorted(self.target_manager.active_targets.keys())
        if not domains:
            table.add_row(
                "",
                Text("No targets yet - press 'd' to discover one", style="italic"),
                "",
                "",
                "",
                "",
            )

        for idx, domain in enumerate(domains):
            target = self.target_manager.active_targets[domain]
            session_status = str(target.get("session_health", "UNKNOWN"))
            pipeline_status = "ACTIVE" if target.get("pipeline_active") else "INACTIVE"
            last_probe = target.get("last_probe")
            if last_probe:
                probe_status = f"{last_probe['status']} ({last_probe['status_code']})"
            else:
                probe_status = "N/A"

            is_highlighted = idx == self._target_index
            marker = Text("> ", style="bold yellow") if is_highlighted else Text("  ")
            domain_text = Text(
                domain, style="bold cyan" if is_highlighted else "cyan"
            )
            table.add_row(
                marker,
                domain_text,
                str(target.get("provider_family", "")),
                session_status,
                pipeline_status,
                probe_status,
            )

        return Panel(
            table,
            title="[bold]Target Summary[/bold]",
            border_style="bright_blue",
            box=box.ROUNDED,
        )

    def _render_target_details(self, selected_domain: Optional[str]) -> RenderableType:
        """Render the details panel for the selected target."""
        if (
            not selected_domain
            or selected_domain not in self.target_manager.active_targets
        ):
            return Panel(
                Align(
                    Text(
                        "Use Up/Down to highlight a target and Enter to select it",
                        style="italic",
                    ),
                    align="center",
                    vertical="middle",
                ),
                title="[bold]Target Details[/bold]",
                border_style="bright_blue",
                box=box.ROUNDED,
            )

        target = self.target_manager.active_targets[selected_domain]
        target_id = int(target["id"])

        endpoint_table = Table(box=box.SIMPLE, expand=True)
        endpoint_table.add_column("Endpoint Key", style="cyan")
        endpoint_table.add_column("Role", style="green")
        endpoint_table.add_column("Volatility", style="yellow")
        endpoint_table.add_column("Method", style="blue")

        try:
            endpoints = self.db_manager.get_endpoints_by_target_id(target_id)
        except Exception as err:
            logger.error(f"Failed to load endpoints for {selected_domain}: {err}")
            endpoints = []
        if endpoints:
            for ep in endpoints:
                endpoint_table.add_row(
                    str(ep["endpoint_key"]),
                    str(ep["role"]),
                    str(ep["volatility"]),
                    str(ep["method"]),
                )
        else:
            endpoint_table.add_row(
                Text("No endpoints recorded", style="italic"), "", "", ""
            )

        activity_table = Table(box=box.SIMPLE, expand=True)
        activity_table.add_column("Time", style="cyan", no_wrap=True)
        activity_table.add_column("Status", style="green")
        activity_table.add_column("Message", style="yellow")

        try:
            logs = self.db_manager.get_recent_replay_logs(target_id, limit=10)
        except Exception as err:
            logger.error(f"Failed to load replay logs for {selected_domain}: {err}")
            logs = []
        if logs:
            for log in logs:
                activity_table.add_row(
                    str(log["executed_at"]),
                    f"{log['status']} ({log['status_code']})",
                    str(log["message"]),
                )
        else:
            activity_table.add_row(Text("No activity yet", style="italic"), "", "")

        content = Group(
            Text("Endpoints", style="bold"),
            endpoint_table,
            Text("Recent Activity", style="bold"),
            activity_table,
        )
        return Panel(
            content,
            title=f"[bold]Details: {selected_domain}[/bold]",
            border_style="bright_blue",
            box=box.ROUNDED,
        )

    def _render_logs(self) -> RenderableType:
        """Render the live log stream panel."""
        recent_logs = self.log_buffer.get_recent(20)
        log_text = Text()
        for level, message in recent_logs:
            if level == "ERROR":
                log_text.append(f"{level} ", style="bold red")
            elif level in ("WARNING", "WARN"):
                log_text.append(f"{level} ", style="bold yellow")
            elif level == "INFO":
                log_text.append(f"{level} ", style="bold green")
            else:
                log_text.append(f"{level} ", style="bold")
            log_text.append(f"{message}\n")
        if not recent_logs:
            log_text.append("No log messages yet.", style="italic")

        return Panel(
            log_text,
            title="[bold]System Logs[/bold]",
            border_style="bright_blue",
            box=box.ROUNDED,
        )

    def _render_footer(self) -> RenderableType:
        """Render the footer key hints."""
        footer_text = Text.from_markup(
            "[cyan]Up/Down[/cyan] highlight   [cyan]Enter[/cyan] select   "
            "[cyan]d[/cyan] discover   [cyan]t[/cyan] test endpoint   "
            "[cyan]r[/cyan] refresh session   [cyan]p[/cyan] pipeline   "
            "[cyan]x[/cyan] delete target   [cyan]h[/cyan] help   [cyan]q[/cyan] quit"
        )
        return Panel(
            Align(footer_text, align="center"),
            box=box.ROUNDED,
            border_style="bright_blue",
        )

    def _render_help(self) -> RenderableType:
        """Render the help view."""
        help_markup = "\n".join(
            [
                "[bold magenta]Help[/bold magenta]",
                "",
                "[bold]Navigation[/bold]",
                "[cyan]Up / Down[/cyan]   Highlight a target in the summary table",
                "[cyan]Enter[/cyan]         Select the highlighted target",
                "",
                "[bold]Actions[/bold]",
                "[cyan]d[/cyan]   Discover a new target (prompts for a URL)",
                "[cyan]t[/cyan]   Test an endpoint (prompts for domain and endpoint key)",
                "[cyan]r[/cyan]   Refresh the session for a target",
                "[cyan]p[/cyan]   Start or stop the ingestion pipeline for a target",
                "[cyan]x[/cyan]   Delete a target (removes database records and the exported schema)",
                "[cyan]h[/cyan]   Toggle this help view",
                "[cyan]q[/cyan]   Quit the control panel",
                "",
                "[italic]Prompts pause the live display until the answer is submitted.[/italic]",
            ]
        )
        return Panel(
            Align(Text.from_markup(help_markup), align="center", vertical="middle"),
            title="[bold]Help[/bold]",
            border_style="bright_blue",
            box=box.ROUNDED,
        )

    def _refresh_display(self) -> None:
        """Update every layout region with fresh content."""
        self.layout["header"].update(self._render_header())
        self.layout["targets"]["target_summary"].update(self._render_target_summary())
        self.layout["targets"]["target_details"].update(
            self._render_target_details(self.selected_domain)
        )
        if self.show_help:
            self.layout["logs"].update(self._render_help())
        else:
            self.layout["logs"].update(self._render_logs())
        self.layout["footer"].update(self._render_footer())

    def _move_selection(self, delta: int) -> None:
        """Move the target highlight cursor by the given offset."""
        domains = sorted(self.target_manager.active_targets.keys())
        if not domains:
            return
        self._target_index = (self._target_index + delta) % len(domains)

    def _highlighted_domain(self) -> Optional[str]:
        """Return the domain currently highlighted in the summary table."""
        domains = sorted(self.target_manager.active_targets.keys())
        if not domains:
            return None
        index = min(max(self._target_index, 0), len(domains) - 1)
        return domains[index]

    def _launch_action(
        self, action_name: str, coro: Coroutine[Any, Any, Tuple[bool, str]]
    ) -> None:
        """Schedule a control action in the background and log its outcome."""

        async def _runner() -> None:
            try:
                success, message = await coro
            except Exception as err:
                logger.error(f"{action_name} failed: {err}")
                return
            if success:
                logger.info(f"{action_name}: {message}")
            else:
                logger.warning(f"{action_name} failed: {message}")

        self.task_manager.add_task(_runner())

    async def _prompt(self, live: Live, prompt_text: str, default: str = "") -> str:
        """Prompt the operator for a line of input without blocking the event loop.

        The live display and the key listener are paused while the prompt is
        visible so that regular line editing works. Both are resumed once the
        answer has been submitted.

        Args:
            live: The active ``Live`` display to pause and resume.
            prompt_text: The question shown to the operator.
            default: Value returned when the operator submits an empty line.

        Returns:
            The stripped answer, or an empty string when input failed.
        """
        live.stop()
        if self.key_listener is not None:
            self.key_listener.stop()

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[str]" = loop.create_future()

        def _resolve(fut: "asyncio.Future[str]", value: str) -> None:
            if not fut.done():
                fut.set_result(value)

        def _read_line() -> None:
            answer = ""
            try:
                answer = Prompt.ask(
                    prompt_text, default=default, show_default=bool(default)
                )
            except Exception as err:
                logger.error(f"Error while reading prompt input: {err}")
            try:
                loop.call_soon_threadsafe(_resolve, future, answer)
            except RuntimeError:
                # The event loop is already closed during shutdown.
                pass

        try:
            reader = threading.Thread(
                target=_read_line, name="tui-prompt-reader", daemon=True
            )
            reader.start()
            answer = await future
        finally:
            if self.active:
                if self.key_listener is not None:
                    self.key_listener.start()
                live.start()

        return answer.strip()

    async def _handle_input(self, live: Live) -> None:
        """Handle single-keypress commands from the operator."""
        if self.key_listener is None:
            return

        while self.active:
            try:
                key = await self.key_listener.get()
                if not self.active:
                    break

                command = key.lower() if len(key) == 1 else key

                if command in ("q", "\x04"):
                    logger.info("Quit requested. Shutting down the control panel...")
                    self.active = False
                    break

                if command == "h":
                    self.show_help = not self.show_help
                    continue

                if command == "ESC":
                    if self.show_help:
                        self.show_help = False
                    continue

                if command == "UP":
                    self._move_selection(-1)
                    continue

                if command == "DOWN":
                    self._move_selection(1)
                    continue

                if command in ("\r", "\n"):
                    selected = self._highlighted_domain()
                    if selected is not None:
                        self.selected_domain = selected
                        logger.info(f"Target selected: {selected}")
                    continue

                if command == "d":
                    url = await self._prompt(
                        live, "Enter target URL to discover (empty to cancel)"
                    )
                    if url:
                        async def _prompt_wrapper(prompt_text: str) -> str:
                            return await self._prompt(live, prompt_text)

                        self.task_manager.add_task(
                            self.target_manager.discover_new_target(
                                url, prompt_cb=_prompt_wrapper
                            )
                        )
                    continue

                if command == "t":
                    domain = await self._prompt(
                        live, "Enter target domain", default=self.selected_domain or ""
                    )
                    if not domain:
                        continue
                    if domain not in self.target_manager.active_targets:
                        logger.warning(f"Unknown target domain: {domain}")
                        continue
                    endpoint_key = await self._prompt(live, "Enter endpoint key")
                    if not endpoint_key:
                        continue
                    self._launch_action(
                        f"Endpoint test on {domain}",
                        self.target_manager.test_endpoint(domain, endpoint_key),
                    )
                    continue

                if command == "r":
                    domain = await self._prompt(
                        live,
                        "Enter target domain to refresh session",
                        default=self.selected_domain or "",
                    )
                    if not domain:
                        continue
                    if domain not in self.target_manager.active_targets:
                        logger.warning(f"Unknown target domain: {domain}")
                        continue
                    self._launch_action(
                        f"Session refresh on {domain}",
                        self.target_manager.refresh_session(domain),
                    )
                    continue

                if command == "p":
                    domain = await self._prompt(
                        live,
                        "Enter target domain to toggle pipeline",
                        default=self.selected_domain or "",
                    )
                    if not domain:
                        continue
                    if domain not in self.target_manager.active_targets:
                        logger.warning(f"Unknown target domain: {domain}")
                        continue
                    self._launch_action(
                        f"Pipeline toggle on {domain}",
                        self.target_manager.toggle_pipeline(domain),
                    )
                    continue

                # Any other key press is ignored.

            except Exception as err:
                logger.error(f"Error while handling keyboard input: {err}")

    async def _refresh_targets_periodically(self) -> None:
        """Periodically reload targets and refresh their statuses."""
        while self.active:
            try:
                self.target_manager.reload_targets()
                for domain in list(self.target_manager.active_targets.keys()):
                    await self.target_manager.refresh_target_status(domain)
                if (
                    self.selected_domain is not None
                    and self.selected_domain
                    not in self.target_manager.active_targets
                ):
                    self.selected_domain = None
            except Exception as err:
                logger.error(f"Error while refreshing target statuses: {err}")
            await asyncio.sleep(5.0)

    async def run(self) -> None:
        """Run the terminal UI until the operator quits."""
        loop = asyncio.get_running_loop()
        self.key_listener = KeyListener(loop)
        self.key_listener.start()

        live = Live(self.layout, console=self.console, refresh_per_second=10)
        try:
            self._refresh_display()
            live.start()
            logger.info("Control panel started. Press 'h' for help, 'q' to quit.")

            self.task_manager.add_task(self._refresh_targets_periodically())
            self.task_manager.add_task(self._handle_input(live))

            try:
                while self.active:
                    try:
                        self._refresh_display()
                    except Exception as err:
                        logger.error(f"Error while refreshing the display: {err}")
                    await asyncio.sleep(0.25)
            finally:
                # Graceful shutdown of background work while the loop still runs.
                self.active = False
                await self.task_manager.shutdown()
                await self.target_manager.stop_all_pipelines()
                self.db_manager.close()
        finally:
            live.stop()
            self.key_listener.stop()


if __name__ == "__main__":
    try:
        terminal = TUITerminal()
        asyncio.run(terminal.run())
        print("[+] Control panel exited cleanly.")
    except KeyboardInterrupt:
        print("\n[!] Keyboard interrupt received. Exiting.")
    except Exception as err:
        print(f"[-] Fatal error: {err}")
        sys.exit(1)
