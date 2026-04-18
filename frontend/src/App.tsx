import { useEffect, useMemo, useState } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import { BrowserPanel } from "./components/BrowserPanel";
import { ChatPanel } from "./components/ChatPanel";
import { TaskPanel } from "./components/TaskPanel";
import { HumanReviewDialog } from "./components/HumanReviewDialog";
import { Sidebar } from "./components/Sidebar";

type AppMode = "chat" | "task";

function App() {
  const [mode, setMode] = useState<AppMode>("task");

  const sessionId = useMemo(() => {
    const stored = sessionStorage.getItem("dqe-session-id");
    if (stored) return stored;
    const id = `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    sessionStorage.setItem("dqe-session-id", id);
    return id;
  }, []);

  const { status, messages, send } = useWebSocket(sessionId);

  // Auto-switch to Task tab when a plan starts — user must see execution + selection UI
  useEffect(() => {
    const last = messages[messages.length - 1];
    if (last?.type === "plan_created") {
      setMode("task");
    }
  }, [messages]);

  // Detect whether the agent is waiting for the user to interact
  const hasPendingInput = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const t = messages[i].type;
      if (t === "selection_request" || t === "human_review") return true;
      if (t === "agent_done" || t === "error" || t === "workflow_done") return false;
    }
    return false;
  }, [messages]);

  return (
    <div className="h-screen flex flex-col bg-gray-950">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-4 py-0 flex items-center justify-between shrink-0 h-11">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold text-gray-100 tracking-wide">DQE Agent</h1>
          <span className="text-xs bg-gray-800 text-gray-500 rounded px-2 py-0.5">v3.0</span>
        </div>

        {/* Mode tabs — top-level navigation */}
        <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => setMode("chat")}
            className={`relative px-4 py-1 rounded-md text-xs font-medium transition-colors ${
              mode === "chat"
                ? "bg-blue-600 text-white shadow"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            💬 Chat
            {hasPendingInput && mode !== "chat" && (
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-yellow-400 rounded-full animate-ping" />
            )}
          </button>
          <button
            onClick={() => setMode("task")}
            className={`relative px-4 py-1 rounded-md text-xs font-medium transition-colors ${
              mode === "task"
                ? "bg-green-600 text-white shadow"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            🤖 Task
            {hasPendingInput && mode !== "task" && (
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-yellow-400 rounded-full animate-ping" />
            )}
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              status === "connected" ? "bg-green-500" : status === "connecting" ? "bg-yellow-500" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-500">{status}</span>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar */}
        <aside className="w-52 shrink-0 p-2 overflow-y-auto">
          <Sidebar sessionId={sessionId} status={status} messages={messages} />
        </aside>

        {/* Left panel: Chat or Task depending on mode */}
        <main className="w-96 shrink-0 p-2">
          {mode === "chat" ? (
            <ChatPanel messages={messages} send={send} />
          ) : (
            <TaskPanel messages={messages} send={send} />
          )}
        </main>

        {/* Right: live browser */}
        <div className="flex-1 p-2 overflow-hidden">
          <BrowserPanel
            messages={messages}
            send={send}
            viewportWidth={1280}
            viewportHeight={800}
            wsConnected={status === "connected"}
          />
        </div>
      </div>

      {/* Modal overlay */}
      <HumanReviewDialog />
    </div>
  );
}

export default App;
