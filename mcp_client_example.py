import asyncio

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    server_url = "http://localhost:8000/sse"  # adjust if HOST/PORT changed

    # 1) Establish the SSE connection
    async with sse_client(server_url) as (read, write):
        # 2) Create an MCP client session over those streams
        async with ClientSession(read, write) as session:
            # 3) Perform the initialization handshake
            await session.initialize()

            # 4) List available tools to verify the server is responding
            tools_response = await session.list_tools()
            tool_names = [tool.name for tool in tools_response.tools]
            print("Available tools on DKG server:", tool_names)

            # ─────────────────────────────────────────────────────────────────
            # 5a) Call `query_dkg_by_name`
            # ─────────────────────────────────────────────────────────────────
            name_to_search = "Alice"
            query_result = await session.call_tool(
                "query_dkg_by_name",
                {"name": name_to_search}
            )
            # FIX: access .content which is the list of TextContent objects
            print(f"---\nSPARQL results for '{name_to_search}':\n")
            if query_result.content:
                print(query_result.content[0].text)
            else:
                print("(no content returned)")

            # ─────────────────────────────────────────────────────────────────
            # 5b) Call `create_knowledge_asset`
            # ─────────────────────────────────────────────────────────────────
            content_to_publish = (
                "Alice is a researcher at OriginTrail DKG, working on decentralized knowledge graphs."
            )
            create_result = await session.call_tool(
                "create_knowledge_asset",
                {"content": content_to_publish}
            )
            print("\n---\ncreate_knowledge_asset returned:\n")
            if create_result.content:
                print(create_result.content[0].text)
            else:
                print("(no content returned)")

asyncio.run(main())
