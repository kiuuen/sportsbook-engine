#define _WIN32_WINNT 0x0A00
#include <winsock2.h>
#include <windows.h>
#include <iostream>
#include <string>
#include "include/httplib.h"
#include "include/json.hpp"
using json = nlohmann::json;
 
void FetchAndParseChromeTabs() {
    httplib::Client cli("http://localhost:9222");
    auto res = cli.Get("/json");
    if (res && res->status == 200) {
        json tabs = json::parse(res->body);
        std::cout << "\n--- Active Chrome Tabs ---" << std::endl;
        for (const auto& tab : tabs) {
            std::string title = tab.value("title", "Untitled");
            std::string url = tab.value("url", "about:blank");
            std::string wsUrl = tab.value("webSocketDebuggerUrl", "");

            std::cout << "Title: " << title << std::endl;
            std::cout << "URL:   " << url << std::endl;
            std::cout << "WS:    " << wsUrl << "\n------------------------" << std::endl;
        }
    } else {
        std::cerr << "Failed to fetch JSON from Chrome. Is it running?" << std::endl;
    }
}

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
        FetchAndParseChromeTabs();
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