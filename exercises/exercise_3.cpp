/*
 * EXERCISE 3: NON-BLOCKING LIFECYCLE MONITOR (WIN32)
 *
 * Objective:
 * Implement an asynchronous monitoring loop that detects whether Chrome is
 * still running or if the operator manually closed the window, without
 * freezing or blocking your main execution thread.
 *
 * Background:
 * In Win32, process handles act as synchronization objects:
 * - While the process is running, the handle is "nonsignaled".
 * - When the process terminates, the OS kernel transitions the handle to "signaled".
 *
 * Technical Requirements:
 * 1. Launch Chrome silently (reusing your setup from Exercise 2).
 * 2. Create a monitoring loop that ticks every 500ms for a maximum of 30 seconds.
 * 3. Inside each  ntinues immediately.
 *    - Evaluate the return state:
 *      * State A: Process is still executing. Print a heartbeat indicator (e.g., ".")
 *                 and sleep for 500ms before the next tick.
 *      * State B: Process has terminated (e.g., operator clicked the "X" on Chrome).
 *                 Immediately break out of the loop.
 * 4. Post-termination handling:
 *    - If termination was detected, query the kernel for the process's final exit code.
 *    - Print the exit code to stdout.
 * 5. Handle cleanup:
 *    - If the 30-second timeout expires while Chrome is still open, terminate it.
 *    - Ensure pi.hProcess and pi.hThread are closed before exiting main().
 *
 * Win32 APIs & Concepts to Research:
 * - WaitForSingleObject (specifically using a timeout of 0, and handling
 *   WAIT_TIMEOUT vs WAIT_OBJECT_0)
 * - GetExitCodeProcess
 * - STILL_ACTIVE (constant associated with exit codes)
 * - CloseHandle
 *
 * Success Criteria:
 * - Run the program. While the heartbeat dots are printing, manually click the "X" 
 *   to close the Chrome window.
 * - Your terminal must detect the closure within 500ms, stop the loop immediately,
 *   print the captured exit code, and cleanly exit without hanging.
 */

 #include <limits>
#include <thread>
#include <windows.h>
 #include <iostream>
 #include <chrono>
 
int main() {
    STARTUPINFOW siw;
    PROCESS_INFORMATION pi;
    ZeroMemory(&siw, sizeof(siw));
    ZeroMemory(&pi, sizeof(pi));
    siw.cb = sizeof(siw);
    std::wstring chrome_path = L"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
    std::wstring cmd_line = L"\"" + chrome_path + L"\" "
                            L"--remote-debugging-port=9222 "
                            L"--user-data-dir=C:\\temp\\cdp_test_profile "
                            L"--no-first-run "
                            L"--no-default-browser-check"
                            L"--enable-logging";

    BOOL success = CreateProcessW(
        NULL,
        cmd_line.data(),
        NULL,
        NULL,
        TRUE,
        0,
        NULL,
        NULL,
        &siw,
        &pi
    );
    if (!success) {
        std::cerr << "failed to create process, error: " << GetLastError() << std::endl;
        return -1;
    }
    std::cout << "Launching app..." << std::endl;
    DWORD wait_exit; // declared at 0 to use it later
    auto start = std::chrono::steady_clock::now();
    while (true) {
        wait_exit = WaitForSingleObject(pi.hProcess, 0);
        if (wait_exit == WAIT_OBJECT_0) break;
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start
        ).count();
        if (elapsed >= 30) {
            std::cout << "\nTimeout reached. Terminating Chrome..." << std::endl;
            TerminateProcess(pi.hProcess, 1);
            break;
        }
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    DWORD exit_code = GetExitCodeProcess(pi.hProcess, &exit_code);
    std::cout << "\nProgram closed with exit code: " << exit_code << std::endl;
    TerminateProcess(pi.hProcess, exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return 0;
}