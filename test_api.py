"""Quick test script for the Chat API.

Run this after starting the API server to test basic functionality.
"""

import asyncio
import httpx


async def test_api():
    """Test basic API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing DQE Chat API...\n")
    
    async with httpx.AsyncClient() as client:
        # 1. Health check
        print("1️⃣ Testing health check...")
        resp = await client.get(f"{base_url}/health")
        print(f"   Status: {resp.status_code}")
        print(f"   Result: {resp.json()}\n")
        
        # 2. List tools
        print("2️⃣ Testing tools list...")
        resp = await client.get(f"{base_url}/tools")
        tools = resp.json()["tools"]
        print(f"   Available tools ({len(tools)}): {', '.join(tools[:5])}...\n")
        
        # 3. Simple chat
        print("3️⃣ Testing simple chat...")
        resp = await client.post(
            f"{base_url}/chat",
            json={"message": "Hello! What can you help me with?"}
        )
        result = resp.json()
        print(f"   Response: {result['response'][:100]}...\n")
        
        # 4. Tool call test (Note: this will actually log in!)
        print("4️⃣ Testing tool call (browser_login)...")
        print("   ⚠️  This will actually log in to NetSuite if credentials are configured!")
        choice = input("   Continue? (y/n): ")
        
        if choice.lower() == 'y':
            resp = await client.post(
                f"{base_url}/chat",
                json={"message": "Log in to NetSuite"}
            )
            result = resp.json()
            print(f"   Response: {result['response'][:200]}...")
            if result.get('tool_calls'):
                print(f"   Tools called: {[t['tool'] for t in result['tool_calls']]}")
        else:
            print("   Skipped.\n")
        
        # 5. Reset conversation
        print("\n5️⃣ Testing conversation reset...")
        resp = await client.post(f"{base_url}/reset")
        print(f"   Status: {resp.status_code}")
        print(f"   Result: {resp.json()}\n")
        
    print("✅ All tests complete!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DQE Chat API Test Script")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  uvicorn dqe_agent.api:app --reload --port 8000\n")
    print("="*60 + "\n")
    
    try:
        asyncio.run(test_api())
    except httpx.ConnectError:
        print("\n❌ ERROR: Could not connect to API server at http://localhost:8000")
        print("   Make sure it's running with: uvicorn dqe_agent.api:app --port 8000")
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted.")
