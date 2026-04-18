import React, { useCallback, useEffect, useRef, useState } from "react";
import type { ChatEntry, PlanStep, ServerMessage } from "../types/messages";

interface Props {
  messages: ServerMessage[];
  send: (msg: any) => void;
}

const TASK_TEMPLATES = [
  { label: "Custom task...", value: "" },
  { label: "Search Jira for open bugs", value: "Search Jira for all open bug tickets assigned to me and summarize them" },
  { label: "Create Jira ticket", value: "Create a Jira ticket for the following issue: " },
  { label: "Web research", value: "Search the web for " },
  { label: "Navigate & extract data", value: "Navigate to  and extract the following data: " },
  { label: "NetSuite → CPQ quote", value: "Create a CPQ quote from the NetSuite opportunity with ID: " },
  { label: "Login to a site", value: "Login to " },
];

export function TaskPanel({ messages, send }: Props) {
  const [input, setInput] = useState("");
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [steps, setSteps] = useState<PlanStep[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [awaitingInput, setAwaitingInput] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Track the last processed message index to handle React 18 automatic batching,
  // which may combine multiple rapid setMessages calls into one render cycle.
  // Without this, only the last message in a batch would be processed, causing
  // human_review (and other) messages to be silently skipped.
  const processedIdxRef = useRef(-1);

  useEffect(() => {
    const newMessages = messages.slice(processedIdxRef.current + 1);
    if (newMessages.length === 0) return;
    processedIdxRef.current = messages.length - 1;

    for (const msg of newMessages) {
      const id = `${Date.now()}-${Math.random()}`;
      const ts = Date.now();

      switch (msg.type) {
        case "agent_text":
          setEntries((prev) => [...prev, { id, role: "agent", content: msg.content, timestamp: ts }]);
          break;
        case "tool_start":
          setEntries((prev) => [
            ...prev,
            { id, role: "tool", content: `⚙ ${msg.tool}`, tool: msg.tool, timestamp: ts },
          ]);
          break;
        case "tool_done":
          setEntries((prev) =>
            prev.map((e) =>
              e.tool === msg.tool && e.content === `⚙ ${msg.tool}`
                ? { ...e, content: `✓ ${msg.tool}: ${String(msg.result).slice(0, 120)}` }
                : e
            )
          );
          break;
        case "human_review":
          setEntries((prev) => [...prev, { id, role: "review", content: msg.question, timestamp: ts }]);
          setAwaitingInput(true);
          setIsRunning(false);
          break;
        case "selection_request":
          setEntries((prev) => [...prev, {
            id, role: "selection", content: msg.question, timestamp: ts,
            selectionOptions: msg.options, multiSelect: msg.multi_select,
          }]);
          setAwaitingInput(true);
          setIsRunning(false);
          break;
        case "plan_created":
          setSteps(
            msg.steps.map((s: any) => ({
              id: s.id,
              description: s.description,
              status: "pending" as const,
            }))
          );
          setIsRunning(true);
          break;
        case "step_status":
          setSteps((prev) =>
            prev.map((s) =>
              s.id === msg.step
                ? {
                    ...s,
                    status:
                      msg.status === "success" || msg.status === "done"
                        ? "done"
                        : msg.status === "failed"
                        ? "failed"
                        : "running",
                    tool: msg.tool,
                  }
                : s
            )
          );
          break;
        case "agent_done":
          setIsRunning(false);
          setAwaitingInput(false);
          if (msg.content) {
            setEntries((prev) => [...prev, { id, role: "agent", content: msg.content, timestamp: ts }]);
          }
          break;
        case "error":
          setIsRunning(false);
          setAwaitingInput(false);
          setEntries((prev) => [...prev, { id, role: "system", content: `Error: ${msg.message}`, timestamp: ts }]);
          break;
        case "workflow_done":
          setIsRunning(false);
          setEntries((prev) => [...prev, { id, role: "system", content: `✓ Done: ${msg.summary}`, timestamp: ts }]);
          break;
      }
    }
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  const handleRun = useCallback(() => {
    const text = input.trim();
    if (!text) return;

    setEntries((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: "user", content: text, timestamp: Date.now() },
    ]);

    if (awaitingInput) {
      // Reply to agent question
      send({ type: "human_response", content: text });
      setAwaitingInput(false);
      setIsRunning(true);
    } else {
      // New task
      setSteps([]);
      send({ type: "run_task", task: text });
      setIsRunning(true);
    }
    setInput("");
  }, [input, awaitingInput, isRunning, send]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleRun();
      }
    },
    [handleRun]
  );

  const handleTemplate = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val) setInput(val);
    e.target.value = "";
  }, []);

  const done = steps.filter((s) => s.status === "done").length;
  const running = steps.find((s) => s.status === "running");

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400" />
          <span className="text-sm font-medium text-gray-300">Task</span>
          <span className="text-xs text-gray-600 ml-1">Planner → Executor → Verifier</span>
        </div>
        {isRunning && (
          <span className="text-xs text-green-400 animate-pulse">Running...</span>
        )}
      </div>

      {/* Plan progress (shown when task is running) */}
      {steps.length > 0 && (
        <div className="px-3 py-2 border-b border-gray-800 space-y-1.5">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-500 font-medium">Execution Plan</span>
            <span className="text-xs text-gray-600">{done}/{steps.length}</span>
          </div>
          <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all duration-500"
              style={{ width: `${steps.length ? (done / steps.length) * 100 : 0}%` }}
            />
          </div>
          <div className="space-y-0.5 max-h-28 overflow-y-auto">
            {steps.map((step, i) => (
              <div
                key={step.id}
                className={`flex items-center gap-1.5 text-xs px-1.5 py-0.5 rounded ${
                  step.status === "running"
                    ? "text-blue-300"
                    : step.status === "done"
                    ? "text-green-500"
                    : step.status === "failed"
                    ? "text-red-400"
                    : "text-gray-600"
                }`}
              >
                <span className="w-4 text-center shrink-0">
                  {step.status === "done" ? "✓" : step.status === "failed" ? "✗" : step.status === "running" ? "○" : `${i + 1}`}
                </span>
                <span className="truncate">{step.description}</span>
                {step.status === "running" && step.tool && (
                  <span className="text-gray-600 shrink-0 ml-auto">{step.tool}</span>
                )}
              </div>
            ))}
          </div>
          {running && (
            <div className="text-xs text-blue-400 pl-1">
              Currently: {running.description}
            </div>
          )}
        </div>
      )}

      {/* Log */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {entries.length === 0 && (
          <div className="text-gray-600 text-sm text-center mt-8 space-y-1">
            <div className="text-2xl mb-3">🤖</div>
            <div>Describe a task and the agent will plan and execute it</div>
            <div className="text-xs text-gray-700 mt-2">
              The Planner breaks it into steps, Executor runs each step, Verifier confirms success
            </div>
          </div>
        )}
        {entries.map((e) => (
          e.role === "selection" ? (() => {
            const lastSelId = entries.filter(x => x.role === "selection").at(-1)?.id;
            const isActive = awaitingInput && e.id === lastSelId;
            const sendSelection = (value: string, label: string) => {
              setEntries(prev => [...prev, { id: `user-${Date.now()}`, role: "user", content: label, timestamp: Date.now() }]);
              send({ type: "selection_response", value });
              setAwaitingInput(false);
              setIsRunning(true);
            };
            return (
              <div key={e.id} className="w-full border-2 border-blue-500 bg-blue-950/40 rounded-lg px-4 py-3">
                <div className="text-blue-400 font-bold text-xs uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <span className="text-base">☰</span> Select an option
                </div>
                <p className="text-blue-100 text-sm mb-3">{e.content}</p>
                {isActive && e.selectionOptions && (
                  <div className="flex flex-col gap-2">
                    {e.selectionOptions.map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() => sendSelection(opt.value, opt.label)}
                        className="w-full text-left px-4 py-2.5 bg-blue-800/60 hover:bg-blue-700 border border-blue-600 text-blue-100 text-sm rounded-lg transition-colors cursor-pointer"
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
                {!isActive && <p className="text-blue-400/60 text-xs mt-1">Selection completed</p>}
              </div>
            );
          })() :
          e.role === "review" ? (
            <div key={e.id} className="w-full border-2 border-yellow-500 bg-yellow-950/40 rounded-lg px-4 py-3">
              <div className="text-yellow-400 font-bold text-xs uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <span className="animate-pulse text-base">⚠</span> Action Required — Reply to continue
              </div>
              <pre className="text-yellow-100 text-sm whitespace-pre-wrap font-sans">{e.content}</pre>
            </div>
          ) : (
          <div
            key={e.id}
            className={`text-sm px-3 py-2 rounded-lg max-w-[95%] ${
              e.role === "user"
                ? "bg-green-700/40 text-green-100 ml-auto border border-green-700/50"
                : e.role === "agent"
                ? "bg-gray-800 text-gray-200"
                : e.role === "tool"
                ? "bg-gray-800/40 text-gray-500 text-xs font-mono border border-gray-800"
                : "bg-gray-800 text-gray-400 border border-gray-700"
            }`}
          >
            {e.content}
          </div>
          )
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-gray-800 space-y-2">
        {/* Template picker */}
        <select
          onChange={handleTemplate}
          defaultValue=""
          className="w-full bg-gray-800 text-gray-400 text-xs rounded-lg px-3 py-1.5 outline-none focus:ring-1 focus:ring-green-600"
        >
          <option value="" disabled>Quick templates...</option>
          {TASK_TEMPLATES.filter((t) => t.value).map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        {awaitingInput && (
          <div className="text-xs text-yellow-500 mb-1 flex items-center gap-1">
            <span className="animate-pulse">●</span> Agent is waiting for your answer
          </div>
        )}
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={awaitingInput ? "Type your answer..." : "Describe a task for the agent to plan and execute..."}
            rows={2}
            className={`flex-1 bg-gray-800 text-gray-100 rounded-lg px-3 py-2 text-sm outline-none resize-none focus:ring-1 ${
              awaitingInput ? "ring-1 ring-yellow-500 focus:ring-yellow-400" : "focus:ring-green-500"
            }`}
          />
          <button
            onClick={handleRun}
            disabled={(isRunning && !awaitingInput) || !input.trim()}
            className={`disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm font-medium self-end ${
              awaitingInput ? "bg-yellow-600 hover:bg-yellow-500" : "bg-green-600 hover:bg-green-500"
            }`}
          >
            {awaitingInput ? "Reply" : isRunning ? "Running" : "Run"}
          </button>
        </div>
        <div className="text-xs text-gray-700">
          Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}
