# mcp_server/dkg_mcp_server.py

import os
import sys
import json
import logging
import argparse

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider
from dkg.constants import BlockchainIds

from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse, StreamingResponse

import uvicorn
from logging.handlers import RotatingFileHandler

# ──────────────────────────────────────────────────────────────────────────────
# Logging Setup (File + STDERR)
# ──────────────────────────────────────────────────────────────────────────────
log_file_path = os.path.join(os.path.dirname(__file__), 'debug.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            log_file_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
        ),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("dkg_mpc_server")

# ──────────────────────────────────────────────────────────────────────────────
# MCP Server Initialization
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP("OriginTrail DKG Tools")

# ──────────────────────────────────────────────────────────────────────────────
# Load .env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

required_env = ["ORIGINTRAIL_NODE_URL", "PRIVATE_KEY", "GOOGLE_API_KEY", "BLOCKCHAIN"]
for var in required_env:
    if not os.getenv(var):
        raise ValueError(f"{var} environment variable is required. Please add it to your .env file.")

# ──────────────────────────────────────────────────────────────────────────────
# Initialize DKG Providers
# ──────────────────────────────────────────────────────────────────────────────
node_provider = NodeHTTPProvider(
    endpoint_uri=os.getenv("ORIGINTRAIL_NODE_URL"),
    api_version="v1",
)

blockchain_id = os.getenv("BLOCKCHAIN")
blockchain_provider = BlockchainProvider(
    getattr(BlockchainIds, blockchain_id).value
)

config = {
    "max_number_of_retries": 300,
    "frequency": 2,
}

dkg = DKG(node_provider, blockchain_provider, config)

# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions: Load prompt templates
# ──────────────────────────────────────────────────────────────────────────────
prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')

def load_prompt_template(filename: str) -> PromptTemplate:
    with open(os.path.join(prompts_dir, filename), 'r') as f:
        return PromptTemplate.from_template(f.read())

# ──────────────────────────────────────────────────────────────────────────────
# Setup Google Generative AI LLM
# ──────────────────────────────────────────────────────────────────────────────
def setup_llm() -> ChatGoogleGenerativeAI:
    model = os.getenv("GOOGLE_LLM", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(temperature=0, model=model, max_retries=2)

# ──────────────────────────────────────────────────────────────────────────────
# JSON‐LD Cleaning Helpers
# ──────────────────────────────────────────────────────────────────────────────
def clean_llm_output(output):
    if isinstance(output, AIMessage):
        output = output.content.strip()
    if isinstance(output, str):
        if output.startswith("```") and output.endswith("```"):
            first_newline = output.find("\n")
            if first_newline != -1:
                output = output[first_newline + 1:-3]
            else:
                output = output[3:-3]
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}. Returning empty dict.")
            return {}
    return {}

def create_jsonld_chain(content: str, llm: ChatGoogleGenerativeAI) -> RunnableSequence:
    # 1) Load prompt templates
    system_prompt = load_prompt_template('system_prompt.txt').template
    create_jsonld_text = load_prompt_template('create_jsonld.txt').template
    review_jsonld_text = load_prompt_template('review_jsonld.txt').template

    step1 = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", create_jsonld_text),
    ])

    step2 = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", review_jsonld_text),
    ])

    def prepare_review_input(inputs):
        return {
            "initial_content": content,
            "proposed_jsonld": json.dumps(inputs["proposed_jsonld"])
        }

    chain1 = step1.partial(initial_content=content) | llm | RunnableLambda(lambda x: {"proposed_jsonld": clean_llm_output(x)})
    wrapper = RunnableLambda(prepare_review_input)
    chain2 = step2 | llm | RunnableLambda(lambda x: {"final_jsonld": clean_llm_output(x)})

    return RunnableSequence(chain1, wrapper, chain2)

# ──────────────────────────────────────────────────────────────────────────────
# MCP Tool: query_dkg_by_name
# ──────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def query_dkg_by_name(name: str, ctx: Context = None) -> str:
    """
    SPARQL query: SELECT entities with schema:name matching `name`.
    Returns JSON-serialized results or an error message.
    """
    try:
        logger.info(f"Query DKG tool called with name: {name}")
        sparql_query = f"""
        PREFIX schema: <http://schema.org/>
        SELECT ?s ?name ?description
        WHERE {{
            ?s schema:name ?name ;
               schema:description ?description .
            FILTER(REGEX(?name, "{name}", "i"))
        }}
        LIMIT 5
        """
        logger.debug(f"Executing SPARQL: {sparql_query}")
        query_result = dkg.graph.query(query=sparql_query)
        return f"Query results for '{name}':\n\n{json.dumps(query_result, indent=2)}"
    except Exception as e:
        logger.error(f"Error executing SPARQL query: {e}")
        return f"Error executing SPARQL query: {e}"

# ──────────────────────────────────────────────────────────────────────────────
# MCP Tool: create_knowledge_asset
# ──────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def create_knowledge_asset(content: str, ctx: Context = None) -> str:
    """
    Convert `content` (natural language) → JSON-LD via LangChain → publish to DKG.
    Returns a summary of the operation (UAL, status, finality).
    """
    try:
        logger.info(f"Creating knowledge asset from content (first 100 chars): {content[:100]}...")

        # 1) Setup LLM
        llm = setup_llm()
        # 2) Build chain
        chain = create_jsonld_chain(content, llm)
        # 3) Invoke chain
        result = chain.invoke({})
        if isinstance(result, list):
            result = result[0] if result else {}
        content_dict = result.get("final_jsonld", {})

        # 4) Publish to DKG
        logger.info("Publishing to DKG...")
        create_asset_result = dkg.asset.create(
            content=content_dict,
            options={
                "epochs_num": 2,
                "minimum_number_of_finalization_confirmations": 3,
                "minimum_number_of_node_replications": 1
            },
        )
        ual = create_asset_result.get("UAL", "Unknown")
        publish_status = create_asset_result.get("operation", {}).get("publish", {}).get("status", "Unknown")
        finality = create_asset_result.get("operation", {}).get("finality", {}).get("status", "Unknown")

        response = (
            f"Knowledge Asset created!\n\n"
            f"UAL: {ual}\n"
            f"DKG Explorer Link: https://dkg-testnet.origintrail.io/explore?ual={ual}\n"
            f"Publishing status: {publish_status}\n"
            f"Finality status: {finality}"
        )
        logger.info(f"Asset creation success: UAL={ual}")
        return response

    except Exception as e:
        logger.error(f"Exception in create_knowledge_asset: {e}", exc_info=True)
        return f"Error creating knowledge asset: {e}"

# ──────────────────────────────────────────────────────────────────────────────
# Middleware: Fix SSE Endpoint URLs for Copilot Studio
# ──────────────────────────────────────────────────────────────────────────────
class FixSSEEndpointMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        logger.debug(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        ctype = response.headers.get("content-type", "")
        if ctype.startswith("text/event-stream"):
            async def event_stream():
                host = request.headers.get("host", "127.0.0.1:8000")
                async for chunk in response.body_iterator:
                    text = chunk.decode("utf-8")
                    if "event: endpoint" in text and "data: /" in text:
                        # Replace relative path data: /xxx with full URL
                        start = text.find("data: ")
                        if start != -1:
                            start += 6
                            end = text.find("\r\n", start)
                            if end == -1:
                                end = len(text)
                            rel = text[start:end]
                            full = f"http://{host}{rel}"
                            new_text = text.replace(f"data: {rel}", f"data: {full}")
                            yield new_text.encode("utf-8")
                        else:
                            yield chunk
                    else:
                        yield chunk
            return StreamingResponse(event_stream(), status_code=response.status_code, headers=dict(response.headers))
        return response

# ──────────────────────────────────────────────────────────────────────────────
# Debug endpoint (just returns a simple text)
# ──────────────────────────────────────────────────────────────────────────────
async def debug_endpoint(request):
    return PlainTextResponse("Debug endpoint OK")

# ──────────────────────────────────────────────────────────────────────────────
# Main: Run the MCP server in chosen transport (stdio | sse)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server with a transport.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        required=True,
        help="Transport mode: 'stdio' for local clients, 'sse' for server deployment.",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        logger.info("Starting MCP server with stdio transport")
        mcp.run(transport="stdio")

    elif args.transport == "sse":
        logger.info("Starting MCP server with SSE transport")
        app = Starlette(
            routes=[
                Route("/debug", debug_endpoint),
                Mount("/", app=mcp.sse_app()),
            ]
        )
        app.add_middleware(FixSSEEndpointMiddleware)
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        logger.info(f"SSE server listening on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
