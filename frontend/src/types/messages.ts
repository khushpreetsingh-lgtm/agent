/* WebSocket message types — matches api.py protocol exactly */

// ── Client → Server ────────────────────────────────────────────────────────

export type ClientMessage =
  | { type: "ping" }
  | { type: "chat"; content: string; flow?: string }
  | { type: "run_task"; task: string }
  | { type: "run_workflow"; workflow: string; inputs: Record<string, unknown> }
  | { type: "human_response"; content: string }
  | { type: "webrtc_offer"; sdp: string }
  | {
      type: "browser_input";
      action: string;
      x?: number;
      y?: number;
      text?: string;
      key?: string;
      delta?: number;
    };

// ── Server → Client ────────────────────────────────────────────────────────

export interface ConnectedMsg {
  type: "connected";
  session_id: string;
}
export interface AgentTextMsg {
  type: "agent_text";
  content: string;
}
export interface ToolStartMsg {
  type: "tool_start";
  tool: string;
  args: Record<string, unknown>;
}
export interface ToolDoneMsg {
  type: "tool_done";
  tool: string;
  result: string;
}
export interface AgentDoneMsg {
  type: "agent_done";
  content: string;
}
export interface BrowserFrameMsg {
  type: "browser_frame";
  data: string;
  width: number;
  height: number;
  mime: string;
}
export interface PlanCreatedMsg {
  type: "plan_created";
  steps: { id: string; description: string }[];
}
export interface StepStatusMsg {
  type: "step_status";
  step: string;
  tool?: string;
  status: string;
  result?: string;
}
export interface HumanReviewMsg {
  type: "human_review";
  question: string;
}
export interface SelectionOption {
  value: string;
  label: string;
}
export interface SelectionRequestMsg {
  type: "selection_request";
  question: string;
  options: SelectionOption[];
  multi_select: boolean;
}
export interface WorkflowDoneMsg {
  type: "workflow_done";
  summary: string;
}
export interface ErrorMsg {
  type: "error";
  message: string;
}
export interface PongMsg {
  type: "pong";
}
export interface WebRTCAnswerMsg {
  type: "webrtc_answer";
  sdp: string;
}
export interface ExtractionResultMsg {
  type: "extraction_result";
  step: string;
  data: Record<string, unknown>;
}

export type ServerMessage =
  | ConnectedMsg
  | AgentTextMsg
  | ToolStartMsg
  | ToolDoneMsg
  | AgentDoneMsg
  | BrowserFrameMsg
  | PlanCreatedMsg
  | StepStatusMsg
  | HumanReviewMsg
  | SelectionRequestMsg
  | WorkflowDoneMsg
  | ErrorMsg
  | PongMsg
  | WebRTCAnswerMsg
  | ExtractionResultMsg;

// ── UI State ────────────────────────────────────────────────────────────────

export interface ChatEntry {
  id: string;
  role: "user" | "agent" | "tool" | "system" | "extraction" | "review" | "selection";
  content: string;
  tool?: string;
  timestamp: number;
  selectionOptions?: SelectionOption[];
  multiSelect?: boolean;
}

export interface PlanStep {
  id: string;
  description: string;
  status: "pending" | "running" | "done" | "failed";
  tool?: string;
  result?: string;
}
