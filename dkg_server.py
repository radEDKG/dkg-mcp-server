from mcp.server.fastmcp import FastMCP, Context
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider
from dkg.constants import BlockchainIds
from dotenv import load_dotenv
import json
import os
import sys
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from starlette.routing import Route
from starlette.responses import PlainTextResponse
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import argparse

# Configure logging to write to both file and stderr
log_file_path = os.path.join(os.path.dirname(__file__), 'debug.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Size-based rotating file handler
        RotatingFileHandler(
            log_file_path,
            maxBytes=5*1024*1024,  # 5 MB per file
            backupCount=5,          # Keep 5 backup files
        ),
        # Stream handler for console output
        logging.StreamHandler(sys.stderr)
    ]
)

# Initialize a logger
logger = logging.getLogger("dkg_mpc_server")

# Initialize the MCP server
mcp = FastMCP("OriginTrail DKG Tools")

# Load environment variables
load_dotenv()

# Check if the mandatory environment variables are available
if not os.getenv("ORIGINTRAIL_NODE_URL"):
    raise ValueError("ORIGINTRAIL_NODE_URL environment variable is required. Please add it to your .env file.")
if not os.getenv("PRIVATE_KEY"):
    raise ValueError("PRIVATE_KEY environment variable is required. Please add it to your .env file.")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is required. Please add it to your .env file.")

# Initialize DKG providers
node_provider = NodeHTTPProvider(
    endpoint_uri= os.getenv("ORIGINTRAIL_NODE_URL"),
    api_version="v1",
)

# Get the selected blockchain from the environment variables
blockchain_id = os.getenv("BLOCKCHAIN")

# Initialize the blockchain provider
blockchain_provider = BlockchainProvider(
    getattr(BlockchainIds, blockchain_id).value
)

# Configure DKG instance
config = {
    "max_number_of_retries": 300,
    "frequency": 2,
}

# Create the DKG instance
dkg = DKG(node_provider, blockchain_provider, config)

# Define path to prompts directory
prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')

# Function to load prompt from file
def load_prompt_template(filename):
    with open(os.path.join(prompts_dir, filename), 'r') as f:
        return PromptTemplate.from_template(f.read())

# Function to set up the LLM with Google Generative AI    
def setup_llm():
    model = os.getenv("GOOGLE_LLM", "gemini-2.0-flash")  # Default to "gemini-2.0-flash" if GOOGLE_LLM is not set
    llm = ChatGoogleGenerativeAI(temperature=0, model=model, max_retries=2)
    return llm

# Function to clean and parse LLM output to be a compliant JSON-LD structure
def clean_llm_output(output):
    if isinstance(output, AIMessage):
        output = output.content.strip()  # Extract content and remove whitespace
        
    if isinstance(output, str):
        # Remove Markdown code block markers, including language-specific markers like ```json
        if output.startswith("```") and output.endswith("```"):
            first_newline = output.find("\n")
            if first_newline != -1:
                output = output[first_newline + 1:-3]  # Remove the first line and the ending ```
            else:
                output = output[3:-3]  # Fallback to remove generic markers
            logger.debug("Removed Markdown code block markers")

        # Better JSON error handling
        try:
            output = json.loads(output)  # Convert to Python dict or list
            logger.debug(f"Successfully parsed JSON to {type(output).__name__}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}. Returning empty dict.")
            output = {}
        
    return output

# Function to create a chain using LangChain that will take natural language content and convert it to JSON-LD
def create_jsonld_chain(content, llm):
    # Load prompt templates from files
    try:
        with open(os.path.join(prompts_dir, 'system_prompt.txt'), 'r') as f:
            system_prompt = f.read()
        
        with open(os.path.join(prompts_dir, 'create_jsonld.txt'), 'r') as f:
            create_jsonld_text = f.read()
        
        with open(os.path.join(prompts_dir, 'review_jsonld.txt'), 'r') as f:
            review_jsonld_text = f.read()
    
    except Exception as e:
        logger.error(f"Error loading prompt templates: {str(e)}")
        raise
    
    # Create chat prompt templates; you can adjust the templates in the prompts folder as needed
    logger.debug("Creating chat prompt templates")
    step1_create_jsonld = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", create_jsonld_text)
    ])
    
    step2_review_jsonld = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", review_jsonld_text)
    ])
    logger.debug("Chat prompt templates created successfully")
    
    # Chain 1: Create initial JSON-LD with content
    logger.debug("Creating Chain 1: initial JSON-LD creation")
    chain1 = step1_create_jsonld.partial(initial_content=content) | llm | RunnableLambda(lambda x: {"proposed_jsonld": clean_llm_output(x)})
    logger.debug("Chain 1 created successfully")

    # Chain 2: Review the proposed JSON-LD content
    logger.debug("Creating Chain 2: review and wrap JSON-LD")
    chain2 = step2_review_jsonld | llm | RunnableLambda(lambda x: {"final_jsonld": clean_llm_output(x)})
    logger.debug("Chain 2 created successfully")
    
    # Function to keep content for the next chain
    def prepare_review_input(inputs):
        logger.debug("Preparing review input with proposed JSON-LD")
        try:
            prepared_input = {
                "initial_content": content,
                "proposed_jsonld": json.dumps(inputs["proposed_jsonld"])
            }
            logger.debug(f"Review input prepared: {json.dumps(prepared_input)[:200]}...")
            return prepared_input
        except Exception as e:
            logger.error(f"Error preparing review input: {str(e)}")
            raise
    
    content_wrapper = RunnableLambda(prepare_review_input)
    logger.debug("Content wrapper function created")
    
    # Combine both chains into a full sequence
    logger.debug("Building full chain sequence")
    full_chain = RunnableSequence(
        chain1,
        content_wrapper,
        chain2
    )
    logger.debug("Full chain built successfully")
    
    return full_chain


# Define the tools for the MCP server

# Tool to query the DKG by name, with a simple SPARQL query that takes a name as input, allowing users to find entities in the DKG
# Can be expanded to have more comprehensive, flexible, and LLM-generated SPARQL queries, depending on the use case
@mcp.tool()
async def query_dkg_by_name(name: str, ctx: Context = None) -> str:
    """
    Execute a SPARQL query on OriginTrail DKG to find entities by name.
    
    Args:
        name: The name to search for in the DKG (e.g., "OriginTrail")
    
    Returns:
        Results from the SPARQL query in a readable format
    """
    try:
        # Log the query being executed
        logger.info(f"Query DKG tool called with name: {name}")
        
        # Construct the SPARQL query with the provided name
        sparql_query = f"""
        PREFIX schema: <http://schema.org/>
        SELECT ?s ?name ?description
        WHERE {{
            ?s schema:name ?name ;
               schema:description ?description .
            FILTER(REGEX(?name, "{name}", "i") || REGEX(?description, "{name}", "i"))
        }}
        LIMIT 25
        """
        
        logger.debug(f"Executing SPARQL query: {sparql_query}")
        
        # Execute the query
        query_result = dkg.graph.query(query=sparql_query)
        
        logger.debug(f"Query result type: {type(query_result)}")
        logger.debug(f"Query result: {json.dumps(query_result, indent=2)}")
        
        # Simplify the result processing
        return f"Query results for '{name}':\n\n{json.dumps(query_result, indent=2)}"
    
    except Exception as e:
        error_message = f"Error executing SPARQL query: {str(e)}"
        logger.error(f"Exception occurred: {error_message}")
        logger.error(f"Exception type: {type(e)}")
        return error_message


# Tool to create Knowledge Assets on the OriginTrail DKG from natural language text
# In its current form the tool takes a string of text, converts it to structured JSON-LD using schema.org standards
# This can be expanded and adjusted to include additional ontologies, example JSON-LD structures, etc. depending on the use case 
@mcp.tool()
async def create_knowledge_asset(content: str, ctx: Context = None) -> str:
    """
    Create a new knowledge asset on the OriginTrail DKG from natural language text.
    
    This tool analyzes text content, converts it to structured JSON-LD using schema.org standards,
    and publishes it to the OriginTrail Decentralized Knowledge Graph.
    
    Args:
        content: Text content to be converted to schema.org JSON-LD and published to the DKG.
                This can be an article, description, blog post, or any text about an entity.
    
    Returns:
        Details about the created knowledge asset, including its Unique Asset Locator (UAL).
    """
    try:
        # Log the start of asset creation
        logger.info(f"Creating knowledge asset from content: {content[:100]}...")
       
        # Set up LLM
        logger.debug("Setting up LLM...")
        llm = setup_llm()
        logger.debug("LLM setup complete")
        
        # Create and run the chain
        logger.debug("Creating JSON-LD chain...")
        chain = create_jsonld_chain(content, llm)
        logger.debug("Chain created, invoking chain...")
        
        result = chain.invoke({})
        logger.debug("Chain invocation complete")
        
        # Ensure result is a dictionary
        if isinstance(result, list):
            logger.debug("Result is a list, converting to dictionary if possible")
            result = result[0] if result else {}

        # Extract the final structure
        content_dict = result.get("final_jsonld", {})
        logger.debug(f"Generated content dictionary: {json.dumps(content_dict, indent=2)}")

        # Create the asset on the DKG
        logger.info("Creating asset on DKG...")
        create_asset_result = dkg.asset.create(
            content=content_dict,
            options={
                "epochs_num": 2,
                "minimum_number_of_finalization_confirmations": 3,
                "minimum_number_of_node_replications": 1
            },
        )
        logger.info("DKG asset creation complete")
        
        logger.debug(f"Asset creation result: {json.dumps(create_asset_result, indent=2)}")

        # Extract key information from the result with logging - simplified version
        ual = create_asset_result.get("UAL", "Unknown")
        logger.info(f"Extracted UAL: {ual}")
        
        publish_status = create_asset_result.get("operation", {}).get("publish",{}).get("status","Unknown")
        logger.info(f"Extracted publishing status: {publish_status}")
        
        finality = create_asset_result.get("operation", {}).get("finality",{}).get("status","Unknown")
        logger.info(f"Extracted finality: {finality}")
        
        # Format a simplified response as a string
        response = f"""
Knowledge Asset collection successfully created!

UAL: {ual}
DKG Explorer link: https://dkg-testnet.origintrail.io/explore?ual={ual}
Publishing status: {publish_status}
Finality: {finality}
        """
        logger.info(f"Formatted response: {response}")
        
        # Return the formatted string response
        return response
    
    except Exception as e:
        error_message = f"Error creating knowledge asset: {str(e)}"
        logger.error(f"Exception occurred: {error_message}")
        logger.error(f"Exception type: {type(e)}")
        
        # Include full traceback for debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return error_message

# This middleware fixes the content type and endpoint URL in SSE responses,
# enabling the DKG MCP server to function with agents built in Microsoft Copilot Studio
class FixSSEEndpointMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Log the incoming request
        logger.debug(f"Request received: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log the response status and content-type
        logger.debug(f"Response status: {response.status_code}, content-type: {response.headers.get('content-type')}")
        
        # Only process SSE responses
        if response.headers.get("content-type") in ["text/event-stream", "text/event-stream; charset=utf-8"]:
            # Ensure correct content type
            if response.headers.get("content-type") == "text/event-stream; charset=utf-8":
                old_content_type = response.headers.get("content-type")
                response.headers["content-type"] = "text/event-stream"
                logger.debug(f"Changed content-type from '{old_content_type}' to 'text/event-stream'")
                logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Create a custom streaming response to modify the endpoint
            async def stream_with_modified_endpoint():
                host = request.headers.get("host", "134.122.64.26:8000")
                logger.debug(f"Processing SSE stream with host: {host}")
                
                # Get an async iterator over the response body
                async for chunk in response.body_iterator:
                    chunk_str = chunk.decode("utf-8")
                    
                    # Check if this is an endpoint event with relative path
                    if "event: endpoint" in chunk_str and "data: /" in chunk_str:
                        logger.debug(f"Original chunk: {chunk_str.strip()}")
                        
                        # Find the relative path
                        data_pos = chunk_str.find("data: ")
                        if data_pos != -1:
                            data_pos += 6  # Move past "data: "
                            path_end = chunk_str.find("\r\n", data_pos)
                            if path_end == -1:
                                path_end = len(chunk_str)
                            
                            relative_path = chunk_str[data_pos:path_end]
                            full_url = f"http://{host}{relative_path}"
                            
                            # Replace the relative path with full URL
                            new_chunk = chunk_str.replace(f"data: {relative_path}", f"data: {full_url}")
                            logger.debug(f"Modified endpoint from '{relative_path}' to '{full_url}'")
                            
                            yield new_chunk.encode("utf-8")
                        else:
                            yield chunk
                    else:
                        yield chunk
            
            # Return a new streaming response with the modified content
            from starlette.responses import StreamingResponse
            return StreamingResponse(
                stream_with_modified_endpoint(),
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        
        # Log the final response headers
        return response

async def debug_endpoint(request):
    return PlainTextResponse("Debug endpoint working")

# The DKG MCP server can operate in two modes:
# 1. **Stdio Mode**: Suitable for running on local clients such as VSCode, Claude, or Cursor.
# 2. **SSE Transport Mode**: Designed for deployment on servers or virtual machines, enabling accessibility
# to agents built in frameworks like Microsoft Copilot Studio.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the MCP server with the chosen transport mode."
    )
    parser.add_argument(
        '--transport',
        choices=['stdio', 'sse'],
        required=True,
        help="Transport method to use. Options: 'stdio' for local clients (e.g., VSCode), or 'sse' for server deployment. "
             "Example: python dkg_server.py --transport stdio"
    )
    args = parser.parse_args()

    if args.transport == 'stdio':
        logger.info("Starting MCP with stdio transport")
        logger.info("Logging system initialized")
        mcp.run(transport='stdio')

    elif args.transport == 'sse':
        logger.info("Starting MCP with SSE transport")
        logger.info("Logging system initialized")
        
        # Create an ASGI application for the SSE server, integrating the DKG MCP server and debug endpoints
        app = Starlette(
            routes=[
                Route('/debug', debug_endpoint),
                Mount('/', app=mcp.sse_app()),
            ]
        )
        app.add_middleware(FixSSEEndpointMiddleware) # Add the middleware to fix content type and SSE endpoint URLs as defined above

        # Use environment variables for specific configuration of the SSE server or use defaults
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        
        logger.info(f"Server will run on {host}:{port}")
        
        uvicorn.run(app, host=host, port=port)