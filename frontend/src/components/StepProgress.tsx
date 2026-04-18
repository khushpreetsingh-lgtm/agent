import { useEffect, useState } from "react";
import type { ServerMessage } from "../types/messages";

interface Props {
  messages: ServerMessage[];
}

interface StepInfo {
  step: string;
  tool: string;
  status: string;
  result?: string;
  ts: number;
}

export function StepProgress({ messages }: Props) {
  const [steps, setSteps] = useState<StepInfo[]>([]);

  useEffect(() => {
    const latest = messages[messages.length - 1];
    if (!latest || latest.type !== "step_status") return;

    setSteps((prev) => [
      ...prev,
      {
        step: latest.step,
        tool: latest.tool || "",
        status: latest.status,
        result: latest.result,
        ts: Date.now(),
      },
    ]);
  }, [messages]);

  if (steps.length === 0) return null;

  return (
    <div className="bg-gray-900 rounded-lg p-3">
      <div className="text-sm font-medium text-gray-300 mb-2">Step Log</div>
      <div className="space-y-1 max-h-40 overflow-y-auto">
        {steps.map((s, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                s.status === "success" || s.status === "done"
                  ? "bg-green-500"
                  : s.status === "failed"
                  ? "bg-red-500"
                  : "bg-yellow-500"
              }`}
            />
            <span className="text-gray-400 font-mono">{s.tool || s.step}</span>
            <span className="text-gray-600 truncate">{s.result?.slice(0, 60)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
