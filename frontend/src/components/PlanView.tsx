import { useEffect, useState } from "react";
import type { PlanStep, ServerMessage } from "../types/messages";

interface Props {
  messages: ServerMessage[];
}

export function PlanView({ messages }: Props) {
  const [steps, setSteps] = useState<PlanStep[]>([]);

  useEffect(() => {
    const latest = messages[messages.length - 1];
    if (!latest) return;

    if (latest.type === "plan_created") {
      setSteps(
        latest.steps.map((s) => ({
          id: s.id,
          description: s.description,
          status: "pending" as const,
        }))
      );
    }

    if (latest.type === "step_status") {
      setSteps((prev) =>
        prev.map((s) =>
          s.id === latest.step
            ? {
                ...s,
                status: latest.status === "success" || latest.status === "done" ? "done" : latest.status === "failed" ? "failed" : "running",
                tool: latest.tool,
                result: latest.result,
              }
            : s
        )
      );
    }
  }, [messages]);

  if (steps.length === 0) return null;

  const done = steps.filter((s) => s.status === "done").length;

  return (
    <div className="bg-gray-900 rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">Execution Plan</span>
        <span className="text-xs text-gray-500">
          {done}/{steps.length} steps
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-gray-800 rounded-full mb-3 overflow-hidden">
        <div
          className="h-full bg-green-500 transition-all duration-300"
          style={{ width: `${(done / steps.length) * 100}%` }}
        />
      </div>

      <div className="space-y-1.5">
        {steps.map((step, i) => (
          <div
            key={step.id}
            className={`flex items-start gap-2 text-xs px-2 py-1.5 rounded ${
              step.status === "running"
                ? "bg-blue-900/30 border border-blue-700"
                : step.status === "done"
                ? "bg-green-900/20 text-green-400"
                : step.status === "failed"
                ? "bg-red-900/20 text-red-400"
                : "text-gray-500"
            }`}
          >
            <span className="shrink-0 w-5 text-center">
              {step.status === "done"
                ? "\u2713"
                : step.status === "failed"
                ? "\u2717"
                : step.status === "running"
                ? "\u25CB"
                : `${i + 1}`}
            </span>
            <span className="flex-1">{step.description}</span>
            {step.tool && <span className="text-gray-600 shrink-0">{step.tool}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
