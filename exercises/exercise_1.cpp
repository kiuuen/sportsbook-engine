/*
 * EXERCISE 1: SPAWN CHROMIUM VIA WIN32 API
 *
 * Requirements:
 * 1. Target your local Chrome binary path:
 *    "C:\Program Files\Google\Chrome\Application"
 * 2. Format a command line string containing these flags:
 *    --remote-debugging-port=9222
 *    --user-data-dir=C:\\temp\\cdp_test_profile
 *    --no-first-run
 * 3. Initialize STARTUPINFOW and PROCESS_INFORMATION structures.
 * 4. Launch the browser using CreateProcessW.
 * 5. Extract and print the PID (dwProcessId) and thread ID (dwThreadId).
 * 6. Sleep for 5 seconds, terminate the process via TerminateProcess, 
 *    and close both handles using CloseHandle.
 *
 * Win32 functions to research:
 * - CreateProcessW, TerminateProcess, CloseHandle, Sleep
 *
 * Verification:
 * Check Task Manager while sleeping to verify chrome.exe has the matching PID
 * and that http://localhost:9222 responds in another browser.
 */


#include <thread>
#include <windows.h>
#include <iostream>
#include <string>
#include <chrono>

int main() {
    std::wstring chrome_path = L"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
    std::wstring cmd_line = L"\"" + chrome_path + L"\" "
                            L"--remote-debugging-port=9222 "
                            L"--user-data-dir=C:\\temp\\cdp_test_profile "
                            L"--no-first-run "
                            L"--no-default-browser-check";
    STARTUPINFOW siw;
    PROCESS_INFORMATION pi;
    ZeroMemory(&siw, sizeof(siw));
    siw.cb = sizeof(siw);
    ZeroMemory(&pi, sizeof(pi));
    std::wcout << "launching chrome process with parameters: L" << cmd_line << std::endl;
    BOOL success = CreateProcessW(
        NULL,
        cmd_line.data(),
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        NULL,
        &siw,
        &pi
    );
    if (!success) {
        DWORD error = GetLastError();
        std::cerr << "an error has ocurred: " << error << std::endl;
        return -1;
    }
    std::cout << "process launched"
    << "\nchild process pid: " << pi.dwProcessId 
    << "\nmain thread id: " << pi.dwThreadId 
    << std::endl;
    
    for (int i = 10; i > 0; i--) {
        std::cout << "sleeping thread, " << i << " seconds remaining . ." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    TerminateProcess(pi.hProcess, 0);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return 0;
}