import asyncio
import websockets
import json
import time
import subprocess
import os
import sys
import httpx

sys.stdout.reconfigure(encoding='utf-8')

async def check_health(base_url):
    print("[INFO] Waiting for API to pre-warm MCP tools (can take 10-30s)...")
    async with httpx.AsyncClient(timeout=5) as client:
        for _ in range(30):
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    print("[INFO] API Server is healthy and ready!")
                    return True
            except:
                pass
            time.sleep(2)
    return False

async def run_jira_tests():
    print("[INFO] Starting the backend API server...")
    server_process = subprocess.Popen(
        ["conda", "run", "-n", "mcpenv", "uvicorn", "dqe_agent.api:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )
    
    base_url = "http://localhost:8000"
    ws_url = "ws://localhost:8000/ws/jira_interactive_test"
    
    if not await check_health(base_url):
        print("[ERROR] API Server failed to start in time.")
        server_process.terminate()
        return

    queries = [
        "What sprints are currently active in my Jira project?",
        "Create a new story called 'Implement automatic testing via conversational agent'",
        "Search for the story 'Implement automatic testing via conversational agent' and retrieve its details",
        "Update the description of that story I just created to 'Automated testing reduces manual work'",
        "Create a new task under the current active sprint to 'Review test results'",
        "Delete the story 'Implement automatic testing via conversational agent'"
    ]
    
    report_output = ""
    
    try:
        print("\n[INFO] Connecting to WebSocket and running Jira queries...")
        async with websockets.connect(ws_url) as ws:
            ack = json.loads(await ws.recv())
            print(f"[WS] Connected: {ack}")
            
            for index, query in enumerate(queries, 1):
                msg = f"\n----- QUERY {index}/{len(queries)} -----\nUSER: {query}\n"
                print(msg, end="")
                report_output += msg
                
                await ws.send(json.dumps({"type": "chat", "content": query}))
                
                # Wait for agent_done
                agent_texts = []
                tools_used = []
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                    resp = json.loads(raw)
                    mtype = resp.get("type")
                    
                    if mtype == "agent_text":
                        agent_texts.append(resp.get("content", ""))
                    elif mtype == "tool_start":
                        tools_used.append(resp.get("tool", ""))
                    elif mtype == "agent_done":
                        break
                    elif mtype == "error":
                        err = f"ERROR: {resp.get('message')}\n"
                        print(err)
                        report_output += err
                        break
                
                agent_full_reply = "".join(agent_texts)
                ans = f"AGENT: {agent_full_reply}\n"
                tco = f"TOOLS USED: {list(set(tools_used))}\n" if tools_used else ""
                
                print(ans, end="")
                if tco: print(tco, end="")
                report_output += ans + tco

            print("\n[INFO] Tests completed. Saving exact transcription to test_jira_transcript.txt")
            with open("test_jira_transcript.txt", "w", encoding="utf-8") as f:
                f.write(report_output)
    except Exception as e:
        print(f"[EXCEPTION] {e}")
    finally:
        print("\n[INFO] Shutting down API server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    asyncio.run(run_jira_tests())
