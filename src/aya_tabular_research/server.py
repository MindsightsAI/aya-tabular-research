import logging
from contextlib import asynccontextmanager
from typing import (  # Removed Awaitable; Removed Callable
    AsyncIterator,
)

from mcp.server.fastmcp import FastMCP

# Import the mcp_interface module itself using relative import
from . import mcp_interface as mcp_interface_module

# Imports for integrated components
from .core.state_manager import StateManager  # Removed NotifierCallback import

# from .core.models.research_models import StateChangeNotificationParams # Remove notification model import
from .execution.instruction_builder import DirectiveBuilder  # Renamed class
from .execution.report_handler import ReportHandler
from .mcp_interface import register_mcp_handlers
from .planning.planner import Planner
from .storage.knowledge_base import TabularKnowledgeBase
from .storage.tabular_store import TabularStore
from .utils.logging_config import setup_logging

# Define logger at module level
logger = logging.getLogger(__name__)

# --- Constants ---
SERVER_NAME = "aya-guided-inquiry-server"
SERVER_VERSION = "0.4.3"  # Incremented version for status resource approach


# --- Application Context ---
# Holds shared application resources
class AppContext:
    def __init__(self):  # Removed mcp_notifier
        # Order matters: DataStore -> KB -> StateManager -> Planner/Executors
        self.data_store = TabularStore()
        self.knowledge_base = TabularKnowledgeBase(self.data_store)
        self.state_manager = StateManager(self.knowledge_base)  # Removed notifier arg
        self.planner = Planner(self.knowledge_base)
        self.directive_builder = DirectiveBuilder(self.knowledge_base)  # Renamed class
        self.report_handler = ReportHandler(self.knowledge_base, self.state_manager)
        logger.info(
            "AppContext initialized with Phase 3 components and original notifier."
        )


# --- Server Lifespan Management ---
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle: initialize resources, load state, run background tasks, save state."""
    setup_logging()
    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}...")

    # --- Initialize App Context ---
    # Notifier callback removed
    app_context = AppContext()
    # --- Assign components to globals in mcp_interface ---
    mcp_interface_module._state_manager_instance = app_context.state_manager
    mcp_interface_module._planner_instance = app_context.planner
    mcp_interface_module._directive_builder_instance = (
        app_context.directive_builder
    )  # Renamed variable/attribute
    mcp_interface_module._report_handler_instance = app_context.report_handler
    mcp_interface_module._knowledge_base_instance = app_context.knowledge_base
    logger.debug("Assigned component instances to mcp_interface globals.")

    # State is now ephemeral per instance. StateManager initializes to AWAITING_TASK_DEFINITION.
    # KB starts empty. No loading needed.
    logger.info(
        "Server started in stateless mode. Initial state: AWAITING_TASK_DEFINITION."
    )

    # --- Start Background Tasks ---
    # Timeout checker task removed.
    try:
        yield app_context  # Server runs here
    finally:
        # --- Stop Background Tasks ---
        # Timeout checker task removed.
        # No state saving needed in stateless mode.
        logger.info("Server shutting down (stateless mode).")

        logger.info(f"{SERVER_NAME} shutdown complete.")


# --- Create and Configure MCP Server ---
server = FastMCP(
    SERVER_NAME,
    version=SERVER_VERSION,
    lifespan=app_lifespan,
)

# --- Register MCP Handlers ---
register_mcp_handlers(server)

# The server is run via the mcp_server instance.
