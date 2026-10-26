/*
 * EXERCISE 1B: DYNAMIC BROWSER SPAWNER FUNCTION
 *
 * Task:
 * Implement a standalone function from scratch:
 *   bool launch_browser(const std::wstring& binary_path, 
 *                       uint16_t port, 
 *                       const std::wstring& profile_name, 
 *                       PROCESS_INFORMATION& out_pi);
 *
 * Requirements inside launch_browser:
 * 1. Build a dynamic command-line string incorporating:
 *    - --remote-debugging-port=<port>
 *    - --user-data-dir=C:\temp\<profile_name>
 *    - --no-first-run
 *    - --window-size=1200,800
 * 2. Zero-out STARTUPINFOW and PROCESS_INFORMATION, and set si.cb = sizeof(si).
 * 3. Call CreateProcessW ensuring the command-line buffer is mutable.
 * 4. If CreateProcessW fails, print GetLastError() to std::cerr and return false.
 * 5. If successful, assign the resulting structs to out_pi and return true.
 *
 * Requirements in main():
 * 1. Call launch_browser using port 9333 and profile name L"session_test".
 * 2. Print the child PID (dwProcessId) and thread ID (dwThreadId).
 * 3. Sleep for 7 seconds.
 * 4. Kill the process using TerminateProcess.
 * 5. Clean up by closing both hProcess and hThread with CloseHandle.
 *
 * Verification:
 * While the 7-second sleep is active, 'netstat -ano | findstr 9333' must show
 * the port LISTENING, and http://localhost:9333/json/version must return Chrome's JSON.
 */

#include <windows.h>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>

bool launch_browser(const std::wstring& binary_path,
                    uint16_t port,
                    const std::wstring& profile_name,
                    PROCESS_INFORMATION& pi_output) {
    std::wstring cmd_line = L"\"" + binary_path + L"\" "
                            L"--remote-debugging-port=\"" + std::to_wstring(port) + L"\" "
                            L"--user-data-dir=\"" + profile_name + L"\" "
                            L"--no-first-run"
                            L"--window-size=1200,800";
        STARTUPINFOW siw;
        PROCESS_INFORMATION pi;
        ZeroMemory(&siw, sizeof(siw));
        ZeroMemory(&pi, sizeof(pi));
        siw.cb = sizeof(siw);
        bool success = CreateProcessW(
            binary_path.c_str(),
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
            std::cerr << "an error ocurred while launching the browser: " << error << std::endl;
            return false;
        }
        pi_output = pi; 
        return true;
    }

int main() {
    std::wstring PATH = L"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
    uint16_t PORT = 9222;
    char* env_user = std::getenv("USERNAME");
    std::string user_narrow = env_user ? env_user : "DefaultUser";
    std::wstring USER(user_narrow.begin(), user_narrow.end());;
    std::wstring PROFILE_NAME =L"C:\\Users\\" + USER + L"\\AppData\\Local\\Temp\\cdp_test_profile";
    PROCESS_INFORMATION pi;
    if (launch_browser(PATH, PORT, PROFILE_NAME, pi)) {
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
    }

    return 0;
    return 0;
}