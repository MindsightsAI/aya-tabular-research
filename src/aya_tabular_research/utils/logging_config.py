import logging

from .artifact_manager import get_artifact_path

# Define the root logger name for the application
APP_LOGGER_NAME = "aya_tabular_research"


def setup_logging(level=logging.DEBUG):  # Changed default level to DEBUG
    """
    Configures basic logging to a file ('mcp_server.log') for the application logger.
    Removes existing handlers for the specific application logger
    before adding the new one to prevent duplicate messages.
    This avoids interference with MCP stdio communication which uses stdout.
    """
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"
    )
    # Get the specific application logger
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)

    # Remove existing handlers for THIS logger only
    # This prevents interfering with handlers potentially added by MCP itself or other libs
    # to the root logger or their own loggers.
    for handler in app_logger.handlers[:]:  # Iterate over a copy
        app_logger.removeHandler(handler)

    # Add our handler (logging to a file instead of stdout)
    log_file_path = get_artifact_path(filename="mcp_server.log")  # Use artifact manager
    
    file_handler = logging.FileHandler(log_file_path, mode="w")  # Create FileHandler
    file_handler.setFormatter(log_formatter)  # Apply formatter to file_handler
    app_logger.addHandler(file_handler)  # Add file_handler to logger

    # Prevent logs from propagating to the root logger if it has handlers
    # This helps ensure only our intended format goes to stdout via this setup
    # app_logger.propagate = False

    # Optional: Configure level for MCP library logger if needed
    # logging.getLogger("mcp").setLevel(logging.WARNING)


if __name__ == "__main__":
    # Example usage
    setup_logging(logging.DEBUG)
    logger = logging.getLogger(f"{APP_LOGGER_NAME}.test")
    logger.debug("Debug message from app logger")
    logger.info("Info message from app logger")

    # Test propagation (should not appear if root has handlers, e.g. basicConfig was called)
    # logging.basicConfig() # Uncomment to test propagation blocking
    root_logger_test = logging.getLogger(APP_LOGGER_NAME)
    root_logger_test.warning("Warning message via app_logger directly")

    other_logger = logging.getLogger("other_module")
    other_logger.error("Error from other logger (should use root config if any)")
