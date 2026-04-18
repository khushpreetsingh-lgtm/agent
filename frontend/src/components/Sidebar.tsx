import { useEffect, useState } from "react";
import type { ServerMessage } from "../types/messages";
import { fetchFlows, fetchConnectors, fetchTools, resetSession } from "../lib/api";

interface Props {
  sessionId: string;
  status: string;
  messages: ServerMessage[];
}

const MCP_CONNECTORS = [
  { name: "playwright", label: "Playwright", icon: "🌐", desc: "Browser automation" },
  { name: "jira", label: "Jira", icon: "📋", desc: "Issue tracking" },
  { name: "brave-search", label: "Brave Search", icon: "🔍", desc: "Web search" },
];

export function Sidebar({ sessionId, status, messages }: Props) {
  const [flows, setFlows] = useState<string[]>([]);
  const [connectors, setConnectors] = useState<string[]>([]);
  const [toolNames, setToolNames] = useState<string[]>([]);

  useEffect(() => {
    fetchFlows()
      .then((d) => setFlows(d.flows || []))
      .catch(() => {});
    fetchConnectors()
      .then((d) => setConnectors(d.connectors || []))
      .catch(() => {});
    fetchTools()
      .then((d) => setToolNames((d.tools || []).map((t: any) => t.name)))
      .catch(() => {});
  }, []);

  const toolCount = messages.filter((m) => m.type === "tool_start").length;
  const activeSteps = messages.filter((m) => m.type === "step_status" && (m as any).status === "running").length;

  // Determine which MCP connectors are loaded (by checking tool names)
  const activeMcp = MCP_CONNECTORS.filter(
    (mc) => toolNames.some((t) => t.startsWith(mc.name + "_") || t.includes(mc.name))
  );

  return (
    <div className="bg-gray-900 rounded-lg p-3 space-y-4 h-full">
      {/* Session */}
      <div>
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Session</div>
        <div className="text-xs text-gray-600 font-mono break-all leading-relaxed">{sessionId}</div>
        <div className="flex items-center gap-1.5 mt-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              status === "connected" ? "bg-green-500" : status === "connecting" ? "bg-yellow-500" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-500">{status}</span>
        </div>
      </div>

      {/* MCP Connectors */}
      <div>
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">MCP Connectors</div>
        <div className="space-y-1">
          {MCP_CONNECTORS.map((mc) => {
            const isActive = activeMcp.some((a) => a.name === mc.name);
            return (
              <div
                key={mc.name}
                className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs ${
                  isActive ? "bg-gray-800 text-gray-300" : "text-gray-700"
                }`}
              >
                <span>{mc.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className={isActive ? "text-gray-300" : "text-gray-600"}>{mc.label}</div>
                  <div className="text-gray-700 text-[10px]">{mc.desc}</div>
                </div>
                <div className={`w-1.5 h-1.5 rounded-full ${isActive ? "bg-green-500" : "bg-gray-700"}`} />
              </div>
            );
          })}
        </div>
      </div>

      {/* Configured sites (from .env) */}
      {connectors.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Sites</div>
          <div className="space-y-1">
            {connectors.map((c) => (
              <div key={c} className="flex items-center gap-2 px-2 py-1 rounded bg-gray-800 text-xs text-gray-400">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                <span className="capitalize">{c}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Flows */}
      {flows.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Flows</div>
          <div className="space-y-1">
            {flows.map((f) => (
              <div key={f} className="text-xs text-gray-500 bg-gray-800 rounded px-2 py-1">
                {f}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stats */}
      <div>
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Stats</div>
        <div className="text-xs text-gray-600 space-y-0.5">
          <div>Tools called: {toolCount}</div>
          {activeSteps > 0 && <div className="text-green-600">Active steps: {activeSteps}</div>}
        </div>
      </div>

      <div className="pt-1">
        <button
          onClick={() => resetSession(sessionId)}
          className="w-full text-xs bg-gray-800 hover:bg-red-900/40 text-gray-500 hover:text-red-400 rounded-lg px-3 py-2 transition-colors"
        >
          Reset Session
        </button>
      </div>
    </div>
  );
}
