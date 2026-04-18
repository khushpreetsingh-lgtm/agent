const BASE = "";

export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

export async function fetchTools() {
  const res = await fetch(`${BASE}/api/v1/tools`);
  return res.json();
}

export async function fetchFlows() {
  const res = await fetch(`${BASE}/api/v1/flows`);
  return res.json();
}

export async function fetchSessions() {
  const res = await fetch(`${BASE}/api/v1/sessions`);
  return res.json();
}

export async function fetchConnectors() {
  const res = await fetch(`${BASE}/api/v1/connectors`);
  return res.json();
}

export async function fetchTasks(sessionId: string) {
  const res = await fetch(`${BASE}/api/v1/tasks/${sessionId}`);
  return res.json();
}

export async function fetchTraces(limit = 100) {
  const res = await fetch(`${BASE}/api/v1/traces?limit=${limit}`);
  return res.json();
}

export async function resetSession(sessionId: string) {
  const res = await fetch(`${BASE}/api/v1/reset/${sessionId}`, {
    method: "POST",
  });
  return res.json();
}
