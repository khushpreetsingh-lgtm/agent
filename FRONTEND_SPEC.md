# DQE Agent — Frontend WebSocket API Specification
**For the frontend developer. All message types sent/received over WebSocket.**

---

## Connection

```
WebSocket: ws://<host>/ws/<session_id>
```

On connect, server immediately sends:
```json
{"type": "connected", "session_id": "s-abc123"}
```

---

## Messages: Frontend → Backend

### Chat Message
```json
{"type": "chat", "message": "How many issues do I have?"}
```

### Respond to ask_user prompt
```json
{"type": "human_response", "approved": true, "message": "demo sprint"}
```

### Respond to request_selection (button click)
```json
{"type": "selection_response", "selected": "FLAG"}
```
For multi-select:
```json
{"type": "selection_response", "selected": ["FLAG-42", "FLAG-43"]}
```

### Reset session
```
POST /api/v1/reset/<session_id>
```

---

## Messages: Backend → Frontend

### 1. `agent_text` — AI response / completion message
```json
{
  "type": "agent_text",
  "content": "Sprint 'Demo Sprint' created successfully.\n  Name: Demo Sprint\n  Start: 2026-04-22\n  End: 2026-05-06"
}
```
**Render as:** Chat bubble (markdown-aware). This is the primary result the user reads.

---

### 2. `status` — Phase change
```json
{"type": "status", "status": "planning", "message": "Planning your request..."}
{"type": "status", "status": "executing", "message": "Executing step 2/5"}
{"type": "status", "status": "complete", "message": "Done"}
{"type": "status", "status": "failed", "message": "Step failed: ..."}
```
**Render as:** Subtle status bar / spinner overlay. Not a chat message.

---

### 3. `step_status` — Individual step progress
```json
{
  "type": "step_status",
  "step": "create_sprint",
  "tool": "jira_create_sprint",
  "label": "Step 5/5 Create Sprint: success → id: 241",
  "status": "done",
  "result": "{\"id\": 241, \"name\": \"Demo Sprint\", ...}"
}
```
`status` values: `"running"` | `"done"` | `"failed"`

**Render as:** Collapsible step log panel (like GitHub Actions steps). Show spinner while running, green check on done, red X on failed.

---

### 4. `selection_request` — User must pick from a list
```json
{
  "type": "selection_request",
  "question": "Which Jira project should this sprint be created in?",
  "options": [
    {"value": "FLAG", "label": "FLAG — FLAG AI Contract Intelligence"},
    {"value": "GCI",  "label": "GCI — General CI Board"}
  ],
  "multi_select": false
}
```
For multi-select (`multi_select: true`), allow multiple selections with checkboxes.

**Render as:** Button group (single) or checkbox list (multi). Each button sends a `selection_response`.

**CRITICAL:** When rendering option labels as button text, do NOT pre-fill the next input field with the label text. The label is for display only.

---

### 5. `human_review` — ask_user free text prompt
```json
{
  "type": "human_review",
  "question": "What should the sprint name be?"
}
```
**Render as:** Inline text input with Send button. Response goes back as `human_response`.

---

### 6. `browser_frame` — Live browser screenshot
```json
{
  "type": "browser_frame",
  "data": "<base64 PNG>",
  "width": 1280,
  "height": 800,
  "mime": "image/png"
}
```
**Render as:** Live browser view panel (only show when browser tasks are running).

---

### 7. `error`
```json
{"type": "error", "message": "Step failed: Start date cannot be in the past."}
```
**Render as:** Red error banner/toast.

---

## Jira UI Cards — What to Build

The backend returns Jira data as plain text/JSON inside `agent_text`. The frontend should detect and render structured Jira results as rich cards.

### Detection
When `agent_text.content` starts with a known prefix, render as a card:
- Starts with `"You have "` + number → **Count Card**
- Contains list of issues (JSON array with `key`, `summary`, `status` fields) → **Issue List Card**
- Contains sprint data (`startDate`, `endDate`, `state`) → **Sprint Card**
- Starts with `"**Daily Standup Digest**"` → **Standup Card**

Alternatively, the backend can be asked to send a dedicated `jira_data` message type (see below). Coordinate with backend developer if you prefer structured data over parsing.

---

### Count Card
Display when result is a single count number.
```
┌─────────────────────────────────────┐
│  📋  Open Issues Assigned to Me     │
│                                     │
│            14                       │
│         issues                      │
│                                     │
│  [View All Issues]                  │
└─────────────────────────────────────┘
```

### Issue List Card
Display when result is a list of Jira issues.
```
┌──────────────────────────────────────────────────────┐
│  Critical Open Issues (3)                            │
├──────────────────────────────────────────────────────┤
│  FLAG-42  │ Login page crashes on Safari    │ High   │
│  FLAG-39  │ Data export timeout             │ High   │
│  INFRA-12 │ Certificate expiry alert        │ Critical│
└──────────────────────────────────────────────────────┘
```
Each row is clickable (opens Jira issue URL: `<JIRA_URL>/browse/<key>`).

### Sprint Card
```
┌─────────────────────────────────────────────┐
│  Sprint Created: Demo Sprint                │
│  Board: GCI board (simple)                  │
│  Start: Apr 22, 2026  →  End: May 6, 2026  │
│  State: Future                              │
└─────────────────────────────────────────────┘
```

### Standup Digest Card
```
┌──────────────────────────────────────────┐
│  Daily Standup — Apr 20, 2026           │
├────────────┬─────────────────────────────┤
│ In Progress│ FLAG-42 API integration     │
│            │ FLAG-38 Auth refactor        │
├────────────┼─────────────────────────────┤
│ Done       │ FLAG-41 Unit tests          │
├────────────┼─────────────────────────────┤
│ Blockers   │ None                        │
└────────────┴─────────────────────────────┘
```

---

## Workflows — Step Sequences to Expect

### Sprint Creation (5 steps)
1. `selection_request` — pick project
2. *(background step, no UI)*
3. `selection_request` — pick board
4. `human_review` — sprint name
5. *(background step, no UI)* → `agent_text` with sprint details

### Status Transition (2-3 steps)
1. *(background step, no UI)*
2. `selection_request` — pick status (if ambiguous)
3. *(background step, no UI)* → `agent_text` with confirmation

### Add Comment (0-2 steps)
1. `human_review` — ask for comment text (if not given)
2. *(background step)* → `agent_text` with confirmation

### Assign Issue (1-2 steps)
1. *(background user search step)*
2. *(background assign step)* → `agent_text` with confirmation

### Log Time (1-2 steps)
1. `human_review` — ask how many hours (if not given)
2. *(background worklog step)* → `agent_text` with confirmation

---

## Environment Variables Needed on Frontend

```
VITE_WS_URL=ws://localhost:8001
VITE_JIRA_BASE_URL=https://yourcompany.atlassian.net  # for building issue URLs
```

---

## Notes for Frontend Dev

1. **Do not debounce WebSocket messages** — `step_status` messages come fast, buffer and show all.
2. **Markdown rendering** — `agent_text.content` uses `**bold**`, `\n` newlines. Use a markdown renderer.
3. **Selection labels** — The `label` field is for display. The `value` field is what gets sent back. Never use the label as input pre-fill.
4. **Connection drops** — Reconnect automatically with same `session_id`. The backend resumes state.
5. **Session reset** — POST to `/api/v1/reset/<session_id>` before starting a fresh conversation.
