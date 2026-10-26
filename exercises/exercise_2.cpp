/*
 * EXERCISE 2: STANDARD STREAM REDIRECTION (WINDOWS NUL SINK)
 *
 * Objective:
 * Decouple Chromium's diagnostic logging from your terminal by redirecting
 * its standard output (stdout) and standard error (stderr) streams to the
 * Windows null device ("NUL") at process creation time.
 *
 * Technical Requirements:
 * 1. Open a handle to the Windows null sink (L"NUL") with write access.
 * 2. Configure handle inheritance:
 *    - By default, handles opened by a parent process are NOT accessible to children.
 *    - You must ensure the NUL handle is created as inheritable so the child 
 *      process receives permission to write to it.
 * 3. Configure the process startup parameters:
 *    - Instruct Windows to use your custom handles instead of the default console streams.
 *    - Route both stdout and stderr to the inheritable NUL handle.
 *    - Route stdin to an invalid/closed handle so Chrome cannot read terminal input.
 * 4. Launch the process with handle inheritance explicitly enabled.
 * 5. Parent handle hygiene:
 *    - Close your local copy of the NUL handle in the parent immediately after 
 *      spawning (the child retains its own inherited copy in kernel memory).
 * 6. Verification loop:
 *    - Print a clean 1-to-10 counter in your terminal (1 second per tick) while
 *      Chrome is active and loading a website.
 *
 * Win32 APIs & Concepts to Research:
 * - CreateFileW (targeting the "NUL" pseudo-file)
 * - SECURITY_ATTRIBUTES (enabling handle inheritance)
 * - STARTUPINFO / STARTUPINFOW (standard stream redirection fields and flags)
 * - CreateProcessW (enabling the inheritance parameter)
 * - CloseHandle
 *
 * Success Criteria:
 * - Your console displays only your numbers (1 to 10) in a clean line.
 * - Zero diagnostic messages, GPU warnings, or WebRTC logs from Chrome leak into the terminal.
 */

#include <cstddef>
#include <errhandlingapi.h>
#include <handleapi.h>
#include <iostream>
#include <processthreadsapi.h>
#include <synchapi.h>
#include <thread>
#include <windows.h>
#include <string>
#include <chrono>

int main() {
    SECURITY_ATTRIBUTES sa;
    ZeroMemory(&sa, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = true; // child inherit acess
    // all of the child process also share this handle, very useful for chromium for example
    sa.lpSecurityDescriptor = NULL;

    HANDLE h_nul = CreateFileW(
        L"NUL",
        GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        &sa,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    
    if (h_nul == INVALID_HANDLE_VALUE) {
        std::cerr << "error creating the output file, error: " << GetLastError() << std::endl; 
        return -1; 
    }

    STARTUPINFOW siw;
    PROCESS_INFORMATION pi;
    ZeroMemory(&siw, sizeof(siw));
    ZeroMemory(&pi, sizeof(pi));
    siw.cb = sizeof(siw);
    siw.dwFlags |= STARTF_USESTDHANDLES; // ignore the default buffers and use our handle
    siw.hStdOutput = h_nul;
    siw.hStdError = h_nul;
    siw.hStdInput = INVALID_HANDLE_VALUE;
    std::wstring chrome_path = L"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
    std::wstring cmd_line = L"\"" + chrome_path + L"\" "
                            L"--remote-debugging-port=9222 "
                            L"--user-data-dir=C:\\temp\\cdp_test_profile "
                            L"--no-first-run "
                            L"--no-default-browser-check";

    BOOL success = CreateProcessW(
        chrome_path.data(),
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
    CloseHandle(h_nul);
    std::cout << "waiting chrome to open and send all output into the windows nul" << std::endl;
    for (int n = 10; n > 0; n--) {
        std::cout << n << " seconds remaining" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    TerminateProcess(pi.hProcess, 0);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return 0;
}
