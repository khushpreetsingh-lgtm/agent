"""
test_cpq_only.py — Run only the CPQ part using saved opportunity/structure JSON.

Usage:
    python test_cpq_only.py [SESSION_ID]

If SESSION_ID is omitted, defaults to "cpq-test-1".

Steps:
  1. Calls POST /api/v1/preload/{session_id}  — injects output/opportunity.json
                                                and output/structure.json into session memory
  2. Connects via WebSocket
  3. Sends a chat message telling the agent to skip NetSuite and go straight to CPQ
"""

import asyncio
import json
import sys

import httpx
import websockets

API_BASE = "http://localhost:8000"
WS_BASE  = "ws://localhost:8000"


async def run(session_id: str) -> None:
    # ── Step 1: preload saved JSON into session memory ────────────────────────
    print(f"[preload] injecting output/opportunity.json + output/structure.json into session '{session_id}' ...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/api/v1/preload/{session_id}", timeout=10)
        result = resp.json()
        print(f"[preload] {result}")
        if result.get("errors"):
            for err in result["errors"]:
                print(f"  WARNING: {err}")
        if not result.get("loaded"):
            print("ERROR: no data was loaded — make sure output/opportunity.json exists (run the full flow once first)")
            return

    # ── Step 2: connect WebSocket and send CPQ-only instruction ──────────────
    ws_url = f"{WS_BASE}/ws/{session_id}"
    print(f"[ws] connecting to {ws_url} ...")

    async with websockets.connect(ws_url) as ws:
        # Wait for "connected" ack
        ack = json.loads(await ws.recv())
        print(f"[ws] {ack}")

        # Send instruction — data is already in session, agent goes straight to CPQ
        instruction = (
            "The NetSuite data has already been extracted and loaded for this session. "
            "Skip NetSuite entirely. "
            "Go directly to CPQ: login, open the new-quote wizard, fill in all fields "
            "using the opportunity and structure data from this session, and finalize the quote."
        )
        await ws.send(json.dumps({"type": "chat", "content": instruction}))
        print(f"[ws] sent CPQ-only instruction\n{'─'*60}")

        # Stream responses until agent_done
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=300)
            msg = json.loads(raw)
            mtype = msg.get("type", "")

            if mtype == "agent_text":
                print(f"[agent] {msg['content']}")

            elif mtype == "tool_start":
                args = msg.get("args", {})
                print(f"[tool →] {msg['tool']}  args={json.dumps(args)}")

            elif mtype == "tool_done":
                print(f"[tool ✓] {msg['tool']}  result={msg.get('result','')[:120]}")

            elif mtype == "human_review":
                print(f"\n[REVIEW REQUIRED] {msg['question']}")
                answer = input("Your answer: ").strip() or "proceed"
                await ws.send(json.dumps({"type": "human_response", "content": answer}))

            elif mtype == "agent_done":
                print(f"\n{'─'*60}\n[done] {msg.get('content','')}")
                break

            elif mtype == "error":
                print(f"[ERROR] {msg.get('message','')}")
                break

            elif mtype in ("browser_frame", "pong"):
                pass  # skip frames/pings


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "cpq-test-1"
    asyncio.run(run(sid))
