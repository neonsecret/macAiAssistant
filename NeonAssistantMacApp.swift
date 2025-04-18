import SwiftUI
import Foundation
import Combine

@main
struct NeonAssistantMacApp: App {
    @StateObject private var pythonManager = PythonBackendManager()

    var body: some Scene {
        MenuBarExtra {
            ContentView(pythonManager: pythonManager)
                .frame(width: 300, height: 250)
                .background(Color.clear)
               .onAppear {
                DispatchQueue.main.async {
                    if let window = NSApplication.shared.windows.first {
                        window.isOpaque = false
                        window.backgroundColor = .clear
                        window.titlebarAppearsTransparent = true
                        window.styleMask.remove(.titled)
                    }
                }
             }

            Divider()

            Button("Quit") {
                pythonManager.stopPythonBackend()
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
        } label: {
            Image(systemName: "brain.head.profile.fill")
                .resizable()
                .frame(width: 18, height: 18)
                .foregroundColor(.blue)
        }
        .menuBarExtraStyle(.window)
    }
}

class PythonBackendManager: ObservableObject {
//    @Published var lastResponse: String = "No data yet"
    @Published var isBackendRunning: Bool = false
    @Published var isNeonInitialized: Bool = false
    @Published var currentActionText: String = "Waiting for server..."

    private var pythonProcess: Process?
    private var webSocketTask: URLSessionWebSocketTask?
    
    func updateActionText(_ text: String) {
        DispatchQueue.main.async {
            self.currentActionText = text
        }
    }
    
    func startPythonBackend() {
        // Check if process is already running
        if isBackendRunning { return }
        
        if let backendScriptURL = Bundle.main.url(forResource: "backend_server", withExtension: "py", subdirectory: "py_backend"),
           let pythonExecutableURL = Bundle.main.url(forResource: "python", withExtension: nil, subdirectory: "py_backend/bcvenv/bin") {
            let process = Process()
            process.executableURL = pythonExecutableURL
            process.arguments = [backendScriptURL.path]
            process.currentDirectoryPath = backendScriptURL.deletingLastPathComponent().path
            
            do {
                try process.run()
                pythonProcess = process
                isBackendRunning = true
                connectWebSocket()
                print("Python backend started and connection established successfully.")
            } catch {
                print("Failed to start Python backend: \(error)")
            }
        } else {
            print("Failed to locate the backend script or python executable in the bundle.")
        }
    }

    func stopPythonBackend() {
        if let process = pythonProcess, process.isRunning {
            process.terminate()
            print("Python backend stopped")
        }
        isBackendRunning = false
        webSocketTask?.cancel(with: .goingAway, reason: nil)
    }

    private func connectWebSocket() {
        guard let url = URL(string: "ws://127.0.0.1:5372/ws") else {
            print("Invalid WebSocket URL")
            return
        }
        let session = URLSession(configuration: .default)
                webSocketTask = session.webSocketTask(with: url)
                webSocketTask?.resume()
                
                // Use ping to verify the connection.
                webSocketTask?.sendPing { [weak self] error in
                    if let error = error {
                        print("Ping failed: \(error). Retrying in 10 seconds...")
                        self?.webSocketTask?.cancel(with: .goingAway, reason: nil)
                        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
                            self?.connectWebSocket()
                        }
                    } else {
                        print("WebSocket connected successfully")
                        self?.updateActionText("Connected to backend.")
                        self?.listenForMessages()
                    }
                }
    }

    private func listenForMessages() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .failure(let error):
                print("Error receiving WebSocket message: \(error)")
            case .success(let message):
                switch message {
                case .string(let text):
                    DispatchQueue.main.async {
                        // Check for initialization message from NeonAssistant
                        if text == "NeonAssistant initialized" {
                            self?.isNeonInitialized = true
                            print("NeonAssistant is initialized.")
//                            self?.updateActionText("Ready.")
                        }
                        // Check for recording cleanup message
                        else if text == "Recording stopped." {
                            print("Swift: Detected recording cleanup")
                            self?.updateActionText("Question received.")
                        }
                        else{
                            self?.updateActionText(text)
//                            self?.lastResponse = text
                        }
                        print("Swift: received text message: \(text)")
                    }
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        DispatchQueue.main.async {
//                            self?.lastResponse = text
                            print("Received data message: \(text)")
                        }
                    }
                @unknown default:
                    print("Received unknown message type")
                }
                // Continue listening for the next message.
                self?.listenForMessages()
            }
        }
    }
    func sendWebSocketMessage(_ message: String) {
        let wsMessage = URLSessionWebSocketTask.Message.string(message)
        webSocketTask?.send(wsMessage) { error in
            if let error = error {
                print("Error sending WebSocket message: \(error)")
            } else {
                print("Sent WebSocket message: \(message)")
            }
        }
    }
    func sendSplashRequest() {
        guard isBackendRunning else {
            print("Python backend is not running")
            return
        }
        sendWebSocketMessage("check_assistant_running")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {  // check after 3 seconds
            guard self.isNeonInitialized else {
                print("Assistant not ready")
                self.updateActionText("Assistant not ready.")
                return
            }
            self.sendWebSocketMessage("run_query")
        }
//            self?.lastResponse="Asked"

//            if let data = data {
//                do {
//                     Parse JSON response
//                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
//                        // Format the response as a string
//                        let formattedResponse = self?.formatResponse(json) ?? "Invalid response"
//
//                        // Update on main thread
//                        DispatchQueue.main.async {
//                            self?.lastResponse = formattedResponse
//                            print("Splash data received: \(formattedResponse)")
//                        }
//                    }
//                } catch {
//                    print("Error in server call: \(error)")
//                }
//            }
    }
}

class EmptyPythonBackendManager : PythonBackendManager{
    override func startPythonBackend() {}
    override func stopPythonBackend() {}
    override func sendSplashRequest() {}
}
