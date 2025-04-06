import logging

from mcp.server.fastmcp import FastMCP

# Import handlers from the new modules
from .mcp_handlers import (
    prompt_handlers,
    resource_handlers,
    tool_handlers,
)

logger = logging.getLogger(__name__)

# --- Global placeholders removed - using core.instances instead ---


def register_mcp_handlers(mcp_server: FastMCP):
    """Registers MCP tool, resource, and prompt handlers with the FastMCP server instance."""
    logger.info("Registering MCP handlers...")

    # Register Tool Handlers
    mcp_server.tool("research/define_task")(tool_handlers.handle_define_task)
    mcp_server.tool("research/submit_inquiry_report")(
        tool_handlers.handle_submit_inquiry_report
    )
    mcp_server.tool("research/submit_user_clarification")(
        tool_handlers.handle_submit_user_clarification
    )
    mcp_server.tool("research/preview_results")(tool_handlers.handle_preview_results)
    mcp_server.tool("research/export_results")(tool_handlers.handle_export_results)
    logger.debug("Tool handlers registered.")

    # Register Resource Handlers
    mcp_server.resource("research://research/status")(
        resource_handlers.get_server_status
    )
    mcp_server.resource("debug://research/debug_state")(
        resource_handlers.get_debug_state
    )
    logger.debug("Resource handlers registered.")

    # Register Prompt Handlers
    mcp_server.prompt("research/overview")(prompt_handlers.handle_overview)
    logger.debug("Prompt handlers registered.")

    logger.info("MCP handlers registration complete.")


# --- Global Component Setter (REMOVED - Components set directly in server.py) ---
