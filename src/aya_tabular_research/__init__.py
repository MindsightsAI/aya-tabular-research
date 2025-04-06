"""
aya_tabular_research package.
"""

import asyncio
import sys  # Import sys for sys.exit

# Import the configured FastMCP instance directly
from .server import arun, server  # Import arun


def main() -> None:
    """Main entry point for the package.
    Starts the MCP server.
    """
    # Directly run the configured FastMCP server instance
    # The lifespan function within server.py handles initialization
    try:
        asyncio.run(arun())  # Call arun instead of server.run
    except Exception as e:
        # Log critical error if server run fails unexpectedly
        # Ensure logger is configured if running this directly (though usually run via mcp tools)
        import logging

        logger = logging.getLogger(__name__)
        logger.critical(f"Server execution failed: {e}", exc_info=True)
        sys.exit(1)  # Exit with error code


__all__ = ["main", "server"]
