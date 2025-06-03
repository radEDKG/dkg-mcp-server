# OriginTrail DKG MCP Server (example)

## Overview
The OriginTrail DKG MCP Server connects MCP-compatible agents with the OriginTrail Decentralized Knowledge Graph (DKG), making it easy to create, retrieve, link, and exchange verifiable knowledge.

Note: This is BETA software and not recommended for use in production

## Key Features

- **SPARQL Querying**: Retrieve knowledge from the DKG using flexible SPARQL queries.
- **Knowledge Asset Creation**: Convert natural language into structured, schema.org-compliant JSON-LD and publish it to the DKG.
- **Agent Memory**: Store and retrieve decentralized agent memory in a standardized, interoperable way.
- **Interoperability**: Works with any MCP-compatible client, including VS Code, Cursor, Microsoft Copilot agents, and more.

## Getting Started

### 1. Clone the Repository

```sh
git clone <repo-url>
cd otdkg-mcp-server
```

### 2. Install Dependencies

Ensure you have Python 3.10+ installed. Then run:

```sh
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in the required values:

- `ORIGINTRAIL_NODE_URL`: You can use the default public node on testnet, use a different public testnet or mainnet node, or deploy and use your own Edge Node.
- `BLOCKCHAIN`: Blockchain to use for publishing Knowledge Assets on the DKG (e.g., `NEUROWEB_TESTNET`)
- `PRIVATE_KEY`: Private key of the wallet you'll use for publishing Knowledge Assets to the DKG
- `GOOGLE_API_KEY`: API key for Google Generative AI (you can get your API ke at https://aistudio.google.com/)

See `.env.example` for detailed comments and options.

### 4. Run the MCP Server

You can run the server in two modes:

#### a) Stdio Mode (for local clients like VS Code, Cursor, Claude, etc.)

```sh
python dkg_server.py --transport stdio
```

#### b) SSE Mode (for server deployment, making the DKG MCP server accessible to e.g. Microsoft Copilot Studio agents)

```sh
python dkg_server.py --transport sse
```

The SSE server will listen on the configured host and port (see `.env`).

## Usage

Once the server is running, you can import it into your client and gain access to the following out-of-the-box tools:

- **Query the DKG**: Use the `query_dkg_by_name` tool to search for entities by name using SPARQL.
- **Create Knowledge Assets on the DKG**: Use the `create_knowledge_asset` tool to convert natural language into JSON-LD and publish it to the DKG.

These tools are exposed via MCP and can be invoked from any compatible agent or client.

## Compatible Clients

- [x] VS Code
- [x] Cursor
- [x] Claude
- [x] Microsoft Copilot Studio agents
- [x] Any MCP-compatible LLM or agentic framework

## Extending the Server

- **Customize Existing Tools**: Modify and enhance the existing tools in `dkg_server.py` or add new functionality to tailor them to your needs.
- **Add New Tools**: You can easily add new MCP tools by defining new functions in `dkg_server.py` using the `@mcp.tool()` decorator (e.g. a tool that will transform website URLs into knowledge on the DKG).
- **Custom Prompts**: Modify or add prompt templates in the `prompts/` directory to customize LLM behavior.
- **Contribute**: Clone, enhance, and submit pull requests to add new features or tools. Community contributions are welcome!

## Project Structure

- `dkg_server.py` — Main server and tool definitions
- `prompts/` — Prompt templates for LLM-powered tools
- `requirements.txt` — Python dependencies
- `.env.example` — Example environment configuration
- `origintrail-dkg-mcp.yaml` — OpenAPI spec for SSE deployment

---

**Empower your agents to create, retrieve, and exchange verifiable knowledge on OriginTrail DKG!**
