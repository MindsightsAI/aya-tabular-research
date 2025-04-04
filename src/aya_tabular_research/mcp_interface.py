import logging
from typing import TYPE_CHECKING, Optional

from mcp.server.fastmcp import FastMCP

# Import core components needed for setting globals
from .core.state_manager import StateManager
from .execution.instruction_builder import DirectiveBuilder
from .execution.report_handler import ReportHandler

# Import handlers from the new modules
from .mcp_handlers import (
    prompt_handlers,
    resource_handlers,
    tool_handlers,
)
from .planning.planner import Planner
from .storage.knowledge_base import KnowledgeBase

# Use TYPE_CHECKING to import AppContext only for type hinting, avoiding circular import at runtime
if TYPE_CHECKING:
    pass  # No specific type hints needed here currently

logger = logging.getLogger(__name__)

# --- Global placeholders for components (Accessed by handlers) ---
# These remain here so handlers imported from other modules can access them.
# A more robust approach might involve dependency injection during registration,
# but this maintains the current access pattern.
_state_manager_instance: Optional[StateManager] = None
_planner_instance: Optional[Planner] = None
_directive_builder_instance: Optional[DirectiveBuilder] = None
_report_handler_instance: Optional[ReportHandler] = None
_knowledge_base_instance: Optional[KnowledgeBase] = None


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


# --- Global Component Setter (Called during server lifespan) ---


def set_global_components(
    state_manager: StateManager,
    planner: Planner,
    directive_builder: DirectiveBuilder,
    report_handler: ReportHandler,
    knowledge_base: KnowledgeBase,
):
    """Sets the global component instances used by the handlers."""
    global _state_manager_instance, _planner_instance, _directive_builder_instance, _report_handler_instance, _knowledge_base_instance
    _state_manager_instance = state_manager
    _planner_instance = planner
    _directive_builder_instance = directive_builder
    _report_handler_instance = report_handler
    _knowledge_base_instance = knowledge_base
    logger.debug("Global component instances set for MCP interface handlers.")
