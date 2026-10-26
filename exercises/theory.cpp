/* =====================================================================================
 * THEORETICAL SPECIFICATION: MODULE 1 - OPERATING SYSTEM PROCESS CONTROL
 * APPLICATION: Orchestration and Supervision of Chromium Browsers in C++
 * =====================================================================================
 *
 * TABLE OF CONTENTS:
 * 1. Nature of the Process in Modern Operating Systems (Memory and Isolation)
 * 2. Chromium's Internal Multi-Process Architecture
 * 3. Creation Mechanics: The Divide between POSIX and Windows (Win32)
 * 4. The Parent-Child Relationship: Orphan Processes, Zombies, and Descriptors
 * 5. The Chromium Profile Lock Problem (SingletonLock)
 * 6. The Termination Ladder and Signal Management (Graceful vs. Forceful)
 * 7. Redirection and Isolation of Standard Streams (stdout/stderr)
 * 8. Essential Command-Line Arguments for the Passive Ingestion Environment
 * =====================================================================================
 */

/* -------------------------------------------------------------------------------------
 * 1. NATURE OF THE PROCESS IN MODERN OPERATING SYSTEMS
 * -------------------------------------------------------------------------------------
 * In modern Unix-like operating systems (Linux, macOS) and Windows, a process is not
 * merely "executing code"; it is an isolated sandbox that owns:
 *
 *   a) Its own virtual address space (Virtual Memory):
 *      Your C++ program cannot directly read or write to Chrome's memory using standard
 *      pointers. Any attempt to dereference a memory address belonging to another process
 *      triggers a kernel protection fault (Segmentation Fault / Access Violation).
 *      For this reason, communication MUST take place via IPC (Inter-Process Communication),
 *      which in our specific architecture is a local network socket (CDP via WebSocket).
 *
 *   b) A table of file descriptors (File Descriptors in POSIX / Handles in Windows):
 *      Open sockets, on-disk files, and the standard streams (stdin, stdout, stderr)
 *      live in a kernel-managed table associated exclusively with that specific process.
 *
 *   c) Unique Process Identifier (PID):
 *      An integer assigned by the kernel to identify the process globally.
 *      Your C++ application MUST retain Chrome's root PID from the exact millisecond it is
 *      spawned. This PID is the anchor by which you track its lifecycle and enforce cleanup.
 * -------------------------------------------------------------------------------------
 */

/* -------------------------------------------------------------------------------------
 * 2. CHROMIUM'S INTERNAL MULTI-PROCESS ARCHITECTURE
 * -------------------------------------------------------------------------------------
 * A common architectural mistake is assuming Chrome is a single monolithic executable.
 * It is not. Chromium uses a strict multi-process architecture for sandboxing and stability:
 *
 *                 +--------------------------------+
 *                 |   Your C++ CLI Engine          |
 *                 +---------------+----------------+
 *                                 | (Supervises via PID & Socket)
 *                                 v
 *                 +--------------------------------+
 *                 |   Chrome Browser Process       | <--- THIS IS YOUR DIRECT CHILD
 *                 +---------------+----------------+
 *                                 |
 *         +-----------------------+-----------------------+
 *         |                       |                       |
 *         v                       v                       v
 *   [Renderer Process]      [Renderer Process]       [Network Service]
 *   (Tab 1 - JS / DOM)      (Tab 2 - JS / DOM)     (Sockets, TLS, Cache)
 *
 * Critical theoretical implications:
 * 1. When your C++ program launches the Chrome binary, it directly creates ONLY the
 *    "Browser Process" (the root Chrome process).
 * 2. Chrome's Browser Process then spawns its own internal child processes (grandchildren
 *    to your C++ program) to isolate web tabs, rendering engines, and network services.
 * 3. If you terminate Chrome's root process cleanly, it automatically tears down its
 *    child processes. If terminated improperly or abruptly, orphaned Renderer processes
 *    can remain lingering in system RAM, consuming CPU and memory resources.
 * -------------------------------------------------------------------------------------
 */

/* -------------------------------------------------------------------------------------
 * 3. CREATION MECHANICS: THE DIVIDE BETWEEN POSIX AND WINDOWS
 * -------------------------------------------------------------------------------------
 * Standard C++ (up to C++23) lacks a native child-process abstraction in its standard
 * library (unlike threads via <thread>). You must understand how the two dominant OS
 * paradigms handle process creation:
 *
 * A) THE POSIX MODEL (Linux, macOS):
 *    Historically structured around two distinct atomic phases:
 *    1. fork(): Clones the calling process entirely (duplicates page tables, descriptors).
 *    2. execve() / execvp(): Replaces the cloned memory space with the target binary
 *       (loading the Google Chrome executable over the cloned process).
 *
 *    MODERN ALTERNATIVE: Modern systems favor 'posix_spawn' / 'posix_spawnp', which
 *    combines creation and execution into an optimized primitive without unnecessary
 *    virtual memory duplication.
 *
 * B) THE WIN32 MODEL (Windows):
 *    Windows does not use the fork/exec paradigm. It exposes a single consolidated API:
 *    - CreateProcessW(): Accepts the executable path, command-line arguments as a single
 *      flat string, security attributes, and process configuration structs:
 *      * STARTUPINFO: Controls window visibility, standard stream handles, etc.
 *      * PROCESS_INFORMATION: The output struct where the Windows kernel returns the
 *        Process Handle (hProcess) and the numeric process ID (dwProcessId).
 *
 * C) RECOMMENDED ABSTRACTION STRATEGY:
 *    To avoid scattering conditional '#ifdef _WIN32' directives across your business logic,
 *    wrap this into an abstract 'Process' interface with platform-specific implementations
 *    (ProcessPosix and ProcessWin32), or vendor a lean cross-platform library like 'reproc'.
 * -------------------------------------------------------------------------------------
 */

/* -------------------------------------------------------------------------------------
 * 4. THE PARENT-CHILD RELATIONSHIP: ORPHANS, ZOMBIES, AND DESCRIPTORS
 * -------------------------------------------------------------------------------------
 * When your C++ program executes Chrome, your program becomes the PARENT and Chrome
 * becomes the CHILD. The OS kernel enforces strict lifecycle contracts on this link:
 *
 * A) THE ZOMBIE PROCESS PROBLEM (Defunct State in Linux/macOS):
 *    When Chrome exits (for instance, if the operator closes the browser window manually),
 *    the process does NOT immediately vanish from kernel memory.
 *    The kernel retains a zombie process stub holding the exit status until the parent reads it.
 *    - If your C++ program fails to invoke wait() or waitpid(), these zombie entries accumulate
 *      in the kernel's process table.
 *    - Your engine must incorporate a non-blocking reaper loop (or a supervisor thread)
 *      to harvest the child's exit status.
 *
 * B) THE ORPHAN PROCESS PROBLEM:
 *    What happens if your C++ engine crashes or is terminated forcefully while Chrome is running?
 *    - Chrome becomes an "orphan." On POSIX systems, it is readopted by PID 1 (systemd/launchd)
 *      and continues executing silently in the background.
 *    - On Windows, the detached process continues running independently.
 *    - SEVERE SYSTEM IMPACT: The debugging port (e.g., 9222) remains bound by the ghost Chrome
 *      instance. The next time you run your tool, the port will be unavailable, causing
 *      subsequent initialization attempts to fail.
 *
 *    STRUCTURAL MITIGATION:
 *    - On Linux: Invoke prctl(PR_SET_PDEATHSIG, SIGTERM) inside the child process context
 *      so the kernel automatically terminates Chrome if the parent C++ process dies.
 *    - On Windows: Assign the child process to an OS "Job Object" configured with
 *      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. When your C++ handle closes, Windows kills Chrome.
 * -------------------------------------------------------------------------------------
 */

/* -------------------------------------------------------------------------------------
 * 5. THE CHROMIUM PROFILE LOCK PROBLEM (SingletonLock)
 * -------------------------------------------------------------------------------------
 * Chromium enforces strict file-level profile integrity:
 * TWO INSTANCES OF CHROME CANNOT CONCURRENTLY SHARE THE SAME USER DATA DIRECTORY.
 *
 * - When Chrome is launched with a dedicated user directory via:
 *   --user-data-dir=/tmp/cdp_browser_profile
 * - Chrome establishes a physical lock file inside that directory ('SingletonLock' symlink
 *   on Linux/macOS, or a named Mutex on Windows).
 *
 * - Theoretical Significance:
 *   If your C++ engine fails to terminate Chrome cleanly before initiating a new analysis,
 *   the lock file remains pinned on disk. On the next execution, Chrome will refuse to bind
 *   the remote debugging port and immediately terminate with the error:
 *   "The profile appears to be in use by another process."
 *
 * - ARCHITECTURAL REQUIREMENT:
 *   Your process module must verify the absence or clearance of stale lock files,
 *   or ensure isolated, deterministic profile paths per operational session.
 * -------------------------------------------------------------------------------------
 */

/* -------------------------------------------------------------------------------------
 * 6. THE TERMINATION LADDER AND SIGNAL MANAGEMENT (Graceful vs. Forceful)
 * -------------------------------------------------------------------------------------
 * When operator navigation concludes or a timeout occurs, your engine must NEVER resort
 * to forceful termination as its primary action.
 *
 * Implement a sequential "Termination Ladder" (Graceful Degradation):
 *
 *                 +-----------------------------------------------+
 *                 | STAGE 1: Protocol-Level Clean Shutdown       |
 *                 | (Transmit CDP command 'Browser.close' via WS) |
 *                 +-----------------------+-----------------------+
 *                                         | (Terminated in < 2 seconds?)
 *                            YES: EXIT    | NO: Advance down ladder
 *                                         v
 *                 +-----------------------------------------------+
 *                 | STAGE 2: OS-Level Interrupt Signal            |
 *                 | (POSIX: SIGTERM / Windows: WM_CLOSE)          |
 *                 +-----------------------+-----------------------+
 *                                         | (Allows browser to flush storage to disk)
 *                                         | (Terminated in < 3 seconds?)
 *                            YES: EXIT    | NO: Final fallback
 *                                         v
 *                 +-----------------------------------------------+
 *                 | STAGE 3: Forceful Kill                        |
 *                 | (POSIX: SIGKILL / Windows: TerminateProcess)  |
 *                 +-----------------------------------------------+
 *                 (Kernel immediately reclaims memory and severs sockets)
 *
 * Technical Justification:
 * Jumping straight to Stage 3 prevents Chrome from flushing its in-memory cookie jars,
 * local storage, and session caches to the underlying SQLite databases on disk, resulting
 * in partial or total loss of the captured authentication state.
 * -------------------------------------------------------------------------------------
 */

/* -------------------------------------------------------------------------------------
 * 7. REDIRECTION AND ISOLATION OF STANDARD STREAMS (stdout / stderr)
 * -------------------------------------------------------------------------------------
 * When executed directly from the terminal, Chromium is exceptionally verbose. It constantly
 * outputs telemetry, warnings, and internal debug noise across stdout and stderr:
 * - GPU driver fallback notifications.
 * - Audio subsystem warnings (ALSA/PulseAudio/CoreAudio).
 * - WebRTC connection handshakes and extension logging.
 *
 * If your C++ engine launches Chrome without decoupling these streams, Chrome's internal
 * logs will interleave with your application's output, corrupting your terminal control
 * panel, progress indicators, and formatted CLI outputs.
 *
 * Core Concept: Standard stream redirection to /dev/null (POSIX) or NUL (Windows).
 * When configuring child process creation structures, reassign the standard file descriptors:
 *   - Child's stdin  <- Closed or tied to an empty pipe.
 *   - Child's stdout <- Directed to /dev/null or an isolated debug log file.
 *   - Child's stderr <- Directed to /dev/null or an isolated debug log file.
 *
 * This guarantees your C++ application maintains exclusive ownership of the terminal UI.
 * -------------------------------------------------------------------------------------
 */

/* -------------------------------------------------------------------------------------
 * 8. ESSENTIAL COMMAND-LINE ARGUMENTS FOR PASSIVE INGESTION
 * -------------------------------------------------------------------------------------
 * Chrome accepts hundreds of execution flags. For an automated discovery pipeline,
 * you must understand the exact rationale behind these essential switches:
 *
 * 1. --remote-debugging-port=<PORT>
 *    Instructs Chrome to initialize its local HTTP server and WebSocket gateway for CDP.
 *    Without this flag, the browser operates as an inaccessible black box to your C++ engine.
 *
 * 2. --user-data-dir=<PATH>
 *    Forces Chrome to read/write from a specified directory instead of the system default.
 *    This avoids profile collisions with the user's personal browser and isolates captured state.
 *
 * 3. --no-first-run
 *    Suppresses the initial setup wizard ("Welcome to Chrome", "Set as default", "Sign in to Google").
 *    These dialogs trigger extraneous background network traffic that contaminates heuristic analysis.
 *
 * 4. --no-default-browser-check
 *    Suppresses the runtime prompt asking to make Chrome the primary system browser.
 *
 * 5. --disable-background-networking
 *    Disables automated browser updates, extension synchronization, and Google telemetry.
 *    Ensures every intercepted packet originates strictly from the target platform.
 *
 * 6. --headless=new (Conditional flag based on session state)
 *    Enables headless execution for fully automated, background runs when authentication
 *    cookies exist and human interaction is not required.
 * =====================================================================================
 */