# Imports (Lines 1-53, cleaned slightly)
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable, Sequence  # Keep only used types for now

import anyio  # Added anyio back
import mcp.types as types
from mcp.server.lowlevel import Server  # Added NotificationOptions
from mcp.server.lowlevel.helper_types import (  # Corrected import path
    ReadResourceContents,
)
from mcp.server.stdio import stdio_server

# from mcp.shared.context import RequestContext # Removed unused import
from mcp.shared.exceptions import McpError  # Added McpError

# Import specific content types for tool results
from mcp.types import AnyUrl  # Added AnyUrl
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from pydantic import ValidationError as PydanticValidationError

from .core import instances

# Import specific Pydantic models for schema generation
from .core.models.research_models import (
    DefineTaskArgs,
    DefineTaskResult,
    ExportResult,
    ExportResultsArgs,
    SubmitClarificationResult,
    SubmitInquiryReportArgs,
    SubmitReportAndGetDirectiveResult,
    SubmitUserClarificationArgs,
)

# Imports for integrated components
from .core.state_manager import StateManager
from .execution.instruction_builder import DirectiveBuilder
from .execution.report_handler import ReportHandler

# Import handlers directly
from .mcp_handlers import prompt_handlers, resource_handlers, tool_handlers
from .planning.planner import Planner
from .storage.knowledge_base import TabularKnowledgeBase
from .storage.tabular_store import TabularStore
from .utils.logging_config import setup_logging

# Define logger at module level
logger = logging.getLogger(__name__)

# --- Constants ---
SERVER_NAME = "aya-guided-inquiry-server"
SERVER_VERSION = "0.4.3"


# --- Application Context ---
# Holds shared application resources
class AppContext:
    def __init__(self):
        # Read thresholds from environment variables with defaults
        try:
            obstacle_thresh_val = float(os.environ.get("AYA_OBSTACLE_THRESHOLD", "0.5"))
        except ValueError:
            logger.warning("Invalid AYA_OBSTACLE_THRESHOLD env var. Using default 0.5")
            obstacle_thresh_val = 0.5
        try:
            conf_thresh_val = float(os.environ.get("AYA_CONFIDENCE_THRESHOLD", "0.6"))
        except ValueError:
            logger.warning(
                "Invalid AYA_CONFIDENCE_THRESHOLD env var. Using default 0.6"
            )
            conf_thresh_val = 0.6
        try:
            stagnation_cycles_val = int(os.environ.get("AYA_STAGNATION_CYCLES", "3"))
        except ValueError:
            logger.warning("Invalid AYA_STAGNATION_CYCLES env var. Using default 3")
            stagnation_cycles_val = 3

        # Order matters: DataStore -> KB -> StateManager -> Planner/Executors
        self.data_store = TabularStore()
        self.knowledge_base = TabularKnowledgeBase(self.data_store)
        self.state_manager = StateManager(self.knowledge_base)
        # Pass thresholds to Planner
        self.planner = Planner(
            self.knowledge_base,
            obstacle_threshold=obstacle_thresh_val,
            confidence_threshold=conf_thresh_val,
            stagnation_cycles=stagnation_cycles_val,
        )
        self.directive_builder = DirectiveBuilder(self.knowledge_base)
        self.report_handler = ReportHandler(self.knowledge_base, self.state_manager)

        instances.state_manager = self.state_manager
        instances.planner = self.planner
        instances.directive_builder = self.directive_builder
        instances.report_handler = self.report_handler
        instances.knowledge_base = self.knowledge_base
        logger.info("AppContext initialized with components.")


# --- Create MCP Server Instance ---
server = Server(
    SERVER_NAME,
    version=SERVER_VERSION,
    # Lifespan is applied around server.run() later
)


# --- Server Lifespan Management ---
@asynccontextmanager
async def app_lifespan(
    server_instance: Server,  # Renamed arg to avoid conflict
) -> AsyncIterator[AppContext]:
    """Manage application lifecycle: initialize resources, load state, run background tasks, save state."""
    setup_logging()
    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}...")

    # --- Initialize App Context ---
    app_context = AppContext()
    logger.debug("Assigned component instances to core.instances.")

    logger.info(
        "Server started in stateless mode. Initial state: AWAITING_TASK_DEFINITION."
    )

    try:
        yield app_context  # Server runs here
    finally:
        logger.info("Server shutting down (stateless mode).")
        logger.info(f"{SERVER_NAME} shutdown complete.")


# --- Define MCP Handlers via Decorators ---


@server.list_tools()
async def list_server_tools() -> list[types.Tool]:
    """Lists all tools provided by the server with their detailed input schemas."""
    tool_list = [
        types.Tool(
            name="research_define_task",
            description="Starts a new research task. Provide the overall goal, desired data structure, and optional seeds via the `task_definition` argument (see schema). Returns a confirmation (`message`, `summary`) and the *first directive* object (`InstructionObjectV3` or `StrategicReviewDirective` with embedded context) in the `instruction_payload` field to guide your next action. If `instruction_payload` is null, check the `status` field (`clarification_needed` or `research_complete`).",
            inputSchema=DefineTaskArgs.model_json_schema(),
        ),
        types.Tool(
            name="research_submit_inquiry_report",
            description="Submit findings after executing a standard directive (`InstructionObjectV3`) OR submit a strategic decision after analyzing a `StrategicReviewDirective`. Correlate using the `instruction_id` from the directive you processed. Provide findings/decision in the `inquiry_report` argument (see schema). Returns a confirmation (`message`, `summary`) and the *next directive* object (`InstructionObjectV3` or `StrategicReviewDirective` with embedded context) in the `instruction_payload` field. If `instruction_payload` is null, check the `status` field (`clarification_needed` or `research_complete`).",
            inputSchema=SubmitInquiryReportArgs.model_json_schema(),
        ),
        types.Tool(
            name="research_submit_user_clarification",
            description="Provide input/guidance from the human user when the server previously requested clarification. Submit the user's input via the `clarification` argument (see schema). Returns a confirmation (`message`, `summary`), the server's updated status (`new_status`), available tools (`available_tools`), and the *next directive* object (`InstructionObjectV3` or `StrategicReviewDirective` with embedded context) in the `instruction_payload` field. If `instruction_payload` is null, check the `status` field (`clarification_needed` or `research_complete`).",
            inputSchema=SubmitUserClarificationArgs.model_json_schema(),
        ),
        types.Tool(
            name="research_preview_results",
            description="Requests a preview (e.g., first N rows) of the structured data accumulated in the server's knowledge base. Typically used when research is complete. Optionally specify row limit via `limit` argument (see schema). Returns a formatted string (e.g., markdown table) representing the data preview.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Optional number of rows to preview. Defaults to 10.",
                        "default": 10,
                    }
                },
                "required": [],
            },  # Updated schema for optional limit
        ),
        types.Tool(
            name="research_export_results",
            description="Requests a full export of the structured data accumulated in the server's knowledge base, saved to a file on the server. Typically used when research is complete. Specify the desired output format ('csv', 'json', 'parquet') via the `format` argument (see schema). Returns an `ExportResult` object containing the server-side `file_path` of the exported data.",
            inputSchema=ExportResultsArgs.model_json_schema(),
        ),
    ]
    logger.info(f"Listing {len(tool_list)} tools.")
    return tool_list


@server.list_resources()
async def list_server_resources() -> list[types.Resource]:
    """Lists all resources provided by the server."""
    # Define resources based on the URIs handled in handle_read_resource
    resource_list = [
        types.Resource(
            name="research/status",  # Added name derived from URI
            uri="research://research/status",
            description="Provides the current status of the research task and available tools.",
            # Add other metadata if needed, e.g., mime_type="application/json"
        ),
        types.Resource(
            name="research/debug_state",  # Added name derived from URI
            uri="debug://research/debug_state",
            description="FOR DEBUGGING ONLY: Returns a JSON string representation of the current state.",
            # Add other metadata if needed
        ),
    ]
    logger.debug(f"Creating resource list: {[r.model_dump() for r in resource_list]}")
    logger.info(f"Listing {len(resource_list)} resources.")
    return resource_list


@server.list_prompts()
async def list_server_prompts() -> list[types.Prompt]:
    """Lists all prompts provided by the server."""
    # Define prompts based on the names handled in handle_get_prompt
    prompt_list = [
        types.Prompt(
            name="research/overview",
            description="Provides an overview or summary related to the research.",
            # Add inputSchema if the prompt takes arguments
        ),
    ]
    logger.info(f"Listing {len(prompt_list)} prompts.")
    return prompt_list


@server.call_tool()
async def handle_tool_call(
    name: str, arguments: dict  # Removed ctx: RequestContext
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handles incoming tool calls, validates arguments, and dispatches to the correct handler."""
    logger.info(f"Received tool call: {name} with args: {arguments}")

    tool_map = {
        "research_define_task": (tool_handlers.handle_define_task, DefineTaskArgs),
        "research_submit_inquiry_report": (
            tool_handlers.handle_submit_inquiry_report,
            SubmitInquiryReportArgs,
        ),
        "research_submit_user_clarification": (
            tool_handlers.handle_submit_user_clarification,
            SubmitUserClarificationArgs,
        ),
        "research_preview_results": (tool_handlers.handle_preview_results, None),
        "research_export_results": (
            tool_handlers.handle_export_results,
            ExportResultsArgs,
        ),
    }

    if name not in tool_map:
        logger.error(f"Unknown tool called: {name}")
        return [types.TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    handler_func, args_model = tool_map[name]

    try:
        # --- Argument Parsing ---
        if args_model:
            parsed_args = args_model(**arguments)
            # TODO: Address context compatibility if handlers require FastMCP Context methods.
            result = await handler_func(parsed_args)  # Removed ctx
        else:
            result = await handler_func(arguments)  # Removed ctx

        # --- Result Processing ---
        if isinstance(result, str):  # Specific handling for preview_results
            return [types.TextContent(type="text", text=result)]
        elif isinstance(
            result,
            (
                DefineTaskResult,
                SubmitReportAndGetDirectiveResult,
                SubmitClarificationResult,
                ExportResult,
            ),
        ):
            logger.warning(
                f"Serializing complex result object for tool {name} to JSON."
            )
            return [
                types.TextContent(type="text", text=result.model_dump_json(indent=2))
            ]
        elif isinstance(result, Sequence) and all(
            isinstance(item, (TextContent, ImageContent, EmbeddedResource))
            for item in result
        ):
            return result  # Already in correct format
        else:
            logger.error(
                f"Handler for tool {name} returned unexpected type: {type(result)}"
            )
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: Internal server error processing result for tool '{name}'.",
                )
            ]

    except PydanticValidationError as e:
        logger.error(f"Invalid arguments for tool {name}: {e}")
        return [
            types.TextContent(
                type="text",
                text=f"Error: Invalid arguments for tool '{name}':\n{str(e)}",
            )
        ]
    except Exception as e:
        logger.exception(f"Error executing tool {name}: {e}")
        return [
            types.TextContent(
                type="text", text=f"Error executing tool '{name}': {str(e)}"
            )
        ]


@server.read_resource()
async def handle_read_resource(
    uri: AnyUrl,  # Removed ctx: RequestContext
) -> Iterable[ReadResourceContents]:
    """Handles incoming resource read requests and dispatches to the correct handler."""
    uri_str = str(uri)
    # Explicitly log the URI being checked
    logger.info(f"Received resource read request for URI: {uri_str}. Checking against known resources...")

    resource_map = {
        "research://research/status": resource_handlers.get_server_status,
        "debug://research/debug_state": resource_handlers.get_debug_state,
    }
    handler_func = resource_map.get(uri_str)

    if not handler_func:
        logger.error(f"Unknown resource URI requested: {uri_str}")
        raise McpError(
            code=types.METHOD_NOT_FOUND, message=f"Unknown resource URI: {uri_str}"
        )

    try:
        # TODO: Address context compatibility if handlers require FastMCP Context methods.
        result_data = await handler_func()  # Removed ctx

        # Convert result to expected format
        # Assuming handlers return Pydantic models ServerStatus or DebugState
        if isinstance(
            result_data, (resource_handlers.ServerStatus, resource_handlers.DebugState)
        ):
            json_content = result_data.model_dump_json(indent=2)
            logger.debug(f"Returning JSON content for resource {uri_str}")
            return [
                ReadResourceContents(content=json_content, mime_type="application/json")
            ]
        else:
            logger.error(
                f"Handler for resource {uri_str} returned unexpected type: {type(result_data)}"
            )
            raise McpError(
                code=types.INTERNAL_ERROR,
                message=f"Internal error processing resource {uri_str}",
            )

    except Exception as e:
        logger.exception(f"Error reading resource {uri_str}: {e}")
        raise McpError(
            code=types.INTERNAL_ERROR,
            message=f"Error reading resource '{uri_str}': {str(e)}",
        ) from e  # Add 'from e'


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict | None  # Removed ctx: RequestContext
) -> types.GetPromptResult:
    """Handles incoming get prompt requests and dispatches."""
    # Explicitly log the prompt name being checked
    logger.info(f"Received get prompt request for: {name}. Checking against known prompts...")

    prompt_map = {
        "research/overview": prompt_handlers.handle_overview,
    }
    handler_func = prompt_map.get(name)

    if not handler_func:
        logger.error(f"Unknown prompt name requested: {name}")
        raise McpError(code=types.METHOD_NOT_FOUND, message=f"Unknown prompt: {name}")

    try:
        # TODO: Adapt if prompt handlers need specific arguments or FastMCP context.
        result = await handler_func()  # Removed ctx

        if isinstance(result, types.GetPromptResult):
            return result
        else:
            logger.error(
                f"Handler for prompt {name} returned unexpected type: {type(result)}"
            )
            raise McpError(
                code=types.INTERNAL_ERROR,
                message=f"Internal error processing prompt {name}",
            )

    except Exception as e:
        logger.exception(f"Error getting prompt {name}: {e}")
        raise McpError(
            code=types.INTERNAL_ERROR,
            message=f"Error getting prompt '{name}': {str(e)}",
        ) from e  # Add 'from e'


# --- Server Runner ---
async def arun():
    """Asynchronous function to run the MCP server."""
    # Assuming stdio transport for now, adapt if SSE is needed
    logger.info("Starting server with stdio transport...")
    async with stdio_server() as streams:
        logger.debug("Stdio streams acquired.")
        # Wrap the run call with the lifespan manager
        async with app_lifespan(server):  # Pass server instance to lifespan
            logger.debug("Lifespan context entered.")
            # Create initialization options using server capabilities
            # We can define NotificationOptions if we want to support list_changed notifications
            # from mcp.server.lowlevel import NotificationOptions
            # notification_options = NotificationOptions(tools_changed=True) # Example
            init_options = server.create_initialization_options(
                # notification_options=notification_options, # Optional
                # experimental_capabilities={} # Optional
            )
            logger.debug("Initialization options created.")
            await server.run(streams[0], streams[1], init_options)
            logger.debug("Server run finished.")


if __name__ == "__main__":
    logger.info("Starting MCP server directly...")
    try:
        # Setup logging before running
        # setup_logging() # Already called inside app_lifespan
        anyio.run(arun)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as main_err:
        logger.exception(f"Server failed to run: {main_err}")

