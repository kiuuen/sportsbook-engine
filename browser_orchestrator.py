import atexit
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional


class ChromeOrchestrator:

    def __init__(
        self,
        profile_name: str = "default_profile",
        port: int = 9222,
        base_profile_dir: Optional[str] = None,
        headless: bool = False,
    ):
        self.port = port
        self.headless = headless

        # Persistent profile directory configuration
        if base_profile_dir:
            self.profile_dir = pathlib.Path(base_profile_dir) / profile_name
        else:
            self.profile_dir = (
                pathlib.Path(os.environ.get("LOCALAPPDATA", "."))
                / "BrowserOrchestrator"
                / profile_name
            )

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.process: Optional[subprocess.Popen] = None
        atexit.register(self.stop)

    def _find_chrome_executable(self) -> str:
        paths = [
            os.path.expandvars(
                r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"
            ),
            os.path.expandvars(
                r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
            ),
            os.path.expandvars(
                r"%LocalAppData%\Google\Chrome\Application\chrome.exe"
            ),
        ]
        for path in paths:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            "Google Chrome executable was not found in standard Windows paths."
        )

    def start(self, timeout: float = 15.0) -> bool:
        if self.process and self.process.poll() is None:
            return True  # Already running

        chrome_exe = self._find_chrome_executable()

        cmd = [
            chrome_exe,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.profile_dir.resolve()}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
        ]

        if self.headless:
            cmd.append("--headless=new")

        # Start process without popup window on Windows
        creationflags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        # Confirm CDP availability
        if not self._wait_for_cdp(timeout=timeout):
            self.stop()
            raise RuntimeError(
                f"Chrome started, but CDP was not available on port {self.port} after {timeout}s."
            )

        return True

    def _wait_for_cdp(self, timeout: float) -> bool:
        url = f"http://127.0.0.1:{self.port}/json/list"
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                # Process terminated prematurely
                return False
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        if isinstance(data, list):
                            for target in data:
                                if (
                                    target.get("type") == "page"
                                    and "webSocketDebuggerUrl" in target
                                ):
                                    return True
            except (urllib.error.URLError, TimeoutError, ConnectionResetError):
                pass
            time.sleep(0.5)

        return False

    def stop(self) -> None:
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.0)
            self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def get_cdp_url(self) -> str:
        url = f"http://127.0.0.1:{self.port}/json/list"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    if isinstance(data, list):
                        for target in data:
                            if (
                                target.get("type") == "page"
                                and "webSocketDebuggerUrl" in target
                            ):
                                return str(target["webSocketDebuggerUrl"])
        except Exception as err:
            raise RuntimeError(
                f"Failed to obtain CDP WebSocket URL on port {self.port}: {err}"
            ) from err
        raise RuntimeError(
            f"No valid page target found on CDP endpoint on port {self.port}."
        )

if __name__ == "__main__":
    print("[*] Starting Chrome Orchestrator test...")
    orchestrator = ChromeOrchestrator(
        profile_name="target_test_profile", port=9222, headless=False
    )

    try:
        orchestrator.start(timeout=10.0)
        print(
            f"[+] Instance started successfully. CDP active on port {orchestrator.port}."
        )
        print(f"[+] Profile directory: {orchestrator.profile_dir.resolve()}")

        # Verify page target WebSocket URL retrieval
        ws_url = orchestrator.get_cdp_url()
        print(f"[+] Page Target WebSocket URL: {ws_url}")

        time.sleep(3)
    finally:
        print("[*] Stopping Chrome instance...")
        orchestrator.stop()
        print("[+] Process finished and verified.")
