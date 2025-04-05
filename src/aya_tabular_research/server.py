import logging
from contextlib import asynccontextmanager
from typing import (  # Removed Awaitable; Removed Callable
    AsyncIterator,
)

from mcp.server.fastmcp import FastMCP

from .core import instances

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
        self.state_manager = StateManager(self.knowledge_base)
        self.planner = Planner(self.knowledge_base)
        self.directive_builder = DirectiveBuilder(self.knowledge_base)
        self.report_handler = ReportHandler(self.knowledge_base, self.state_manager)

        instances.state_manager = self.state_manager
        instances.planner = self.planner
        instances.directive_builder = self.directive_builder
        instances.report_handler = self.report_handler
        instances.knowledge_base = self.knowledge_base
        logger.info("AppContext initialized with components and original notifier.")


# --- Server Lifespan Management ---
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle: initialize resources, load state, run background tasks, save state."""
    setup_logging()
    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}...")

    # --- Initialize App Context ---
    # Notifier callback removed
    app_context = AppContext()
    logger.debug("Assigned component instances to core.instances.")

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
