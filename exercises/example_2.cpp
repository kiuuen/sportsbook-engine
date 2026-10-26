#include <windows.h>
#include <iostream>
#include <string>

int main() {
    // 1. Security attributes configuration for HANDLE INHERITANCE
    // By default in Windows, handles are NOT inheritable by child processes.
    // We need this structure to explicitly mark the handle as inheritable.
    SECURITY_ATTRIBUTES sa;
    ZeroMemory(&sa, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE; // <--- CRUCIAL: Grants the child process permission to use this handle
    sa.lpSecurityDescriptor = NULL;

    // 2. Open or create the target where we want to send the output
    // In this reference example, we use a real file on disk: "C:\\temp\\cmd_output.log"
    std::wstring log_path = L"C:\\temp\\cmd_output.log";

    HANDLE h_output_file = CreateFileW(
        log_path.c_str(),                         // File path or device name
        GENERIC_WRITE,                            // Desired access: write
        FILE_SHARE_READ | FILE_SHARE_WRITE,       // Sharing mode: allow concurrent reads/writes
        &sa,                                      // Pass attributes with bInheritHandle = TRUE
        CREATE_ALWAYS,                            // Overwrite if exists, create if not
        FILE_ATTRIBUTE_NORMAL,                    // Standard file attributes
        NULL                                      // Template handle
    );

    // Verify the handle is valid before proceeding
    if (h_output_file == INVALID_HANDLE_VALUE) {
        std::cerr << "[-] Failed to create output file. Error: " << GetLastError() << std::endl;
        return 1;
    }

    // 3. Configure STARTUPINFO to override standard streams
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // IMPORTANT:
    // dwFlags must include STARTF_USESTDHANDLES to tell Windows to ignore
    // the default console buffers and use our explicitly assigned handles below.
    si.dwFlags |= STARTF_USESTDHANDLES;

    // Route standard output (stdout) and standard error (stderr) to our file handle
    si.hStdOutput = h_output_file;
    si.hStdError  = h_output_file;

    // Set stdin to INVALID_HANDLE_VALUE so the child process does not wait for keyboard input
    si.hStdInput  = INVALID_HANDLE_VALUE;

    // 4. Command to be executed by the child process
    // Run cmd.exe to list C:\Windows (a command that normally floods the console)
    std::wstring cmd = L"cmd.exe /c dir C:\\Windows";

    std::cout << "[*] Spawning cmd.exe with redirected output..." << std::endl;

    // 5. Create the process with INHERITANCE ENABLED
    BOOL success = CreateProcessW(
        NULL,                 // lpApplicationName
        cmd.data(),           // lpCommandLine (must be a mutable buffer)
        NULL,                 // lpProcessAttributes
        NULL,                 // lpThreadAttributes
        TRUE,                 // <--- CRUCIAL: bInheritHandles = TRUE (allows child to inherit h_output_file)
        0,                    // dwCreationFlags
        NULL,                 // lpEnvironment
        NULL,                 // lpCurrentDirectory
        &si,                  // STARTUPINFO containing redirected handles
        &pi                   // PROCESS_INFORMATION to receive PID and process handle
    );

    // 6. PARENT HANDLE HYGIENE (Critical Win32 detail)
    // Once CreateProcessW succeeds, the child owns its own reference to the handle in kernel memory.
    // The parent MUST close its local copy immediately.
    // If you keep it open, the file stays locked by the parent, and file buffers may not flush cleanly.
    CloseHandle(h_output_file);

    if (!success) {
        std::cerr << "[-] Failed to create process. Error: " << GetLastError() << std::endl;
        return 1;
    }

    // 7. Wait for the child command to finish
    std::cout << "[*] Waiting for the child process to complete silently..." << std::endl;
    WaitForSingleObject(pi.hProcess, INFINITE);

    std::cout << "[+] Child process finished. Notice the terminal remained completely clean." << std::endl;
    std::cout << "[+] Check C:\\temp\\cmd_output.log to verify the captured output." << std::endl;

    // Final cleanup of process and thread handles
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}