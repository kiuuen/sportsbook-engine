#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

int main() {
    // 1. Path to the Chrome executable on Windows
    std::wstring app_path = L"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

    // 2. Build the command-line arguments.
    // NOTE: By Windows convention, the first token should be the application path/name.
    std::wstring cmd_line = L"\"" + app_path + L"\" "
                            L"--remote-debugging-port=9222 "
                            L"--user-data-dir=C:\\temp\\cdp_test_profile "
                            L"--no-first-run "
                            L"--no-default-browser-check";

    // 3. Process configuration structures
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;

    // MANDATORY: Zero the memory of both structures to eliminate garbage data
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si); // 'cb' (Count of Bytes) tells Windows the size/version of the struct
    ZeroMemory(&pi, sizeof(pi));

    std::wcout << L"[*] Launching process with arguments:\n" << cmd_line << L"\n\n";

    // 4. Create the child process
    // IMPORTANT: cmd_line.data() is passed as a mutable buffer because CreateProcessW 
    // may modify the string contents internally while parsing arguments.
    BOOL success = CreateProcessW(
        app_path.c_str(),     // lpApplicationName: Full binary path
        cmd_line.data(),      // lpCommandLine: Complete argument string (must be mutable)
        NULL,                 // lpProcessAttributes: Default security attributes
        NULL,                 // lpThreadAttributes: Default thread security
        FALSE,                // bInheritHandles: Do not inherit parent handles (for now)
        0,                    // dwCreationFlags: No special creation flags
        NULL,                 // lpEnvironment: Inherit parent environment
        NULL,                 // lpCurrentDirectory: Inherit working directory
        &si,                  // lpStartupInfo: Pointer to startup configuration
        &pi                   // lpProcessInformation: Windows writes PIDs and Handles here
    );

    // If CreateProcessW returns FALSE (0), the invocation failed
    if (!success) {
        DWORD error = GetLastError();
        std::cerr << "[-] Failed to spawn process. Win32 error code: " << error << std::endl;
        return 1;
    }

    // 5. The process spawned successfully: harvest PID and Thread ID
    std::cout << "[+] Process launched successfully." << std::endl;
    std::cout << "    Child Process PID: " << pi.dwProcessId << std::endl;
    std::cout << "    Main Thread ID:    " << pi.dwThreadId << std::endl;

    // 6. Wait 5 seconds to verify it is running
    std::cout << "[*] Sleeping for 5 seconds (verify in Task Manager or open localhost:9222)..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 7. Forceful termination (Kill)
    std::cout << "[*] Terminating process..." << std::endl;
    TerminateProcess(pi.hProcess, 0); // 0 is the exit code assigned to the killed process

    // 8. Kernel Handle Cleanup (MANDATORY in Windows)
    // CreateProcessW opens two kernel handles: one for the process and one for the primary thread.
    // If you do not close them with CloseHandle, you leak kernel object table entries.
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    std::cout << "[+] Handles closed. Execution finished." << std::endl;
    return 0;
}