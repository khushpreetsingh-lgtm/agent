import React, { useCallback, useEffect, useRef, useState } from "react";
import type { ChatEntry, ServerMessage } from "../types/messages";

interface Props {
  messages: ServerMessage[];
  send: (msg: any) => void;
}

export function ChatPanel({ messages, send }: Props) {
  const [input, setInput] = useState("");
  const [entries, setEntries] = useState<ChatEntry[]>([]);
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
                ? { ...e, content: `✓ ${msg.tool}` }
                : e
            )
          );
          break;
        case "human_review":
          setEntries((prev) => [...prev, { id, role: "review", content: msg.question, timestamp: ts }]);
          setAwaitingInput(true);
          break;
        case "selection_request":
          setEntries((prev) => [...prev, {
            id, role: "selection", content: msg.question, timestamp: ts,
            selectionOptions: msg.options, multiSelect: msg.multi_select,
          }]);
          setAwaitingInput(true);
          break;
        case "extraction_result": {
          const data = (msg as any).data as Record<string, unknown>;
          const lines = Object.entries(data)
            .filter(([, v]) => v !== null && v !== undefined && v !== "")
            .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`)
            .join("\n");
          setEntries((prev) => [
            ...prev,
            { id, role: "extraction", content: lines, timestamp: ts },
          ]);
          break;
        }
        case "agent_done":
          setAwaitingInput(false);
          if (msg.content) {
            setEntries((prev) => [...prev, { id, role: "agent", content: msg.content, timestamp: ts }]);
          }
          break;
        case "error":
          setAwaitingInput(false);
          setEntries((prev) => [...prev, { id, role: "system", content: `Error: ${msg.message}`, timestamp: ts }]);
          break;
      }
    }
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    setEntries((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: "user", content: text, timestamp: Date.now() },
    ]);
    // If agent is waiting for input, send as human_response; otherwise normal chat
    if (awaitingInput) {
      send({ type: "human_response", content: text });
      setAwaitingInput(false);
    } else {
      send({ type: "chat", content: text });
    }
    setInput("");
  }, [input, awaitingInput, send]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-gray-800 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-blue-400" />
        <span className="text-sm font-medium text-gray-300">Chat</span>
        <span className="text-xs text-gray-600 ml-1">Conversational — ask questions, quick actions</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {entries.length === 0 && (
          <div className="text-gray-600 text-sm text-center mt-8 space-y-1">
            <div className="text-2xl mb-3">💬</div>
            <div>Ask anything or give quick instructions</div>
            <div className="text-xs text-gray-700 mt-2">e.g. "Navigate to google.com" or "What's on this page?"</div>
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
          e.role === "extraction" ? (
            <div key={e.id} className="bg-gray-800 border border-emerald-700/50 rounded-lg px-3 py-2 text-xs font-mono text-gray-300 max-w-[95%]">
              <div className="text-emerald-400 font-semibold mb-1.5 text-xs uppercase tracking-wider">Extracted Data</div>
              {e.content.split("\n").map((line, i) => {
                const colon = line.indexOf(":");
                const key = colon > -1 ? line.slice(0, colon) : line;
                const val = colon > -1 ? line.slice(colon + 1).trim() : "";
                return (
                  <div key={i} className="flex gap-2 py-0.5 border-b border-gray-700/40 last:border-0">
                    <span className="text-gray-500 min-w-[120px] shrink-0">{key}</span>
                    <span className="text-gray-100 break-all">{val}</span>
                  </div>
                );
              })}
            </div>
          ) : e.role === "review" ? (() => {
            // Parse <<<N. Option>>> choice buttons from content
            const choiceRe = /<<<(\d+\.\s*.+?)>>>/g;
            const choices: string[] = [];
            let m: RegExpExecArray | null;
            while ((m = choiceRe.exec(e.content)) !== null) {
              choices.push(m[1].replace(/^\d+\.\s*/, "").trim());
            }
            const cleanContent = e.content.replace(/<<<\d+\.\s*.+?>>>/g, "").trim();
            const lastReviewId = entries.filter(x => x.role === "review").at(-1)?.id;
            const isActive = awaitingInput && e.id === lastReviewId;
            return (
              <div key={e.id} className="w-full border-2 border-yellow-500 bg-yellow-950/40 rounded-lg px-4 py-3">
                <div className="text-yellow-400 font-bold text-xs uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <span className="animate-pulse text-base">⚠</span> Action Required — Reply to continue
                </div>
                <pre className="text-yellow-100 text-sm whitespace-pre-wrap font-sans">{cleanContent}</pre>
                {choices.length > 0 && isActive && (
                  <div className="flex gap-2 mt-3 flex-wrap">
                    {choices.map((choice, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setEntries(prev => [...prev, { id: `user-${Date.now()}`, role: "user", content: choice, timestamp: Date.now() }]);
                          send({ type: "human_response", content: choice });
                          setAwaitingInput(false);
                        }}
                        className="px-4 py-1.5 bg-yellow-500 hover:bg-yellow-400 text-black text-sm font-semibold rounded-lg transition-colors cursor-pointer"
                      >
                        {choice}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })() : (
          <div
            key={e.id}
            className={`text-sm px-3 py-2 rounded-lg max-w-[90%] ${
              e.role === "user"
                ? "bg-blue-600 text-white ml-auto"
                : e.role === "agent"
                ? "bg-gray-800 text-gray-200"
                : e.role === "tool"
                ? "bg-gray-800/50 text-gray-400 text-xs font-mono border border-gray-700"
                : "bg-red-900/30 text-red-300"
            }`}
          >
            {e.content}
          </div>
          )
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-gray-800">
        {awaitingInput && (
          <div className="text-xs text-yellow-500 mb-1 flex items-center gap-1">
            <span className="animate-pulse">●</span> Agent is waiting for your reply
          </div>
        )}
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={awaitingInput ? "Type your answer..." : "Type a message..."}
            className={`flex-1 bg-gray-800 text-gray-100 rounded-lg px-3 py-2 text-sm outline-none focus:ring-1 ${
              awaitingInput ? "ring-1 ring-yellow-500 focus:ring-yellow-400" : "focus:ring-blue-500"
            }`}
          />
          <button
            onClick={handleSend}
            className={`text-white px-4 py-2 rounded-lg text-sm font-medium ${
              awaitingInput ? "bg-yellow-600 hover:bg-yellow-500" : "bg-blue-600 hover:bg-blue-500"
            }`}
          >
            {awaitingInput ? "Reply" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
