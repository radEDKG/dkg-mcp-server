# dkg-mcp-server/mcp_client_example.py

import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    server_url = "http://localhost:8000/sse"

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Confirm the tools exist
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            # 1) Query for anything related to "Quantum Gravity"
            query_result = await session.call_tool(
                "query_dkg_by_name",
                {"name": "Quantum Gravity"}
            )
            print("--- SPARQL results for 'Alice': ---\n")
            print(query_result)

            # 2) (Optional) Create a dummy asset
            create_result = await session.call_tool(
                "create_knowledge_asset",
                {"content": """
                    This knowledge asset is about quantum gravity. 
                    The asset’s name is "QuantumGravityTest1" and its description is 
                    "A simple test asset for quantum gravity on my local Blazegraph node."
                """}
            )
            print("\n--- create_knowledge_asset returned: ---\n")
            print(create_result)

asyncio.run(main())
