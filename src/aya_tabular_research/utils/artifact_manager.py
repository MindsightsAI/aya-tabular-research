import logging
from pathlib import Path

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.debug(f"artifact_manager.py loaded. __file__ = {__file__}")

# Define the base directory for artifacts relative to the project root
# Assuming the project root is the parent of the 'src' directory
# Calculate PROJECT_ROOT with intermediate logging
resolved_path = Path(__file__).resolve()
logger.debug(f"Resolved path: {resolved_path}")
p1 = resolved_path.parent
logger.debug(f"Parent 1: {p1}")
p2 = p1.parent
logger.debug(f"Parent 2: {p2}")
p3 = p2.parent
logger.debug(f"Parent 3: {p3}")
PROJECT_ROOT = p3.parent  # Parent 4
logger.debug(f"Calculated PROJECT_ROOT: {PROJECT_ROOT}")

ARTIFACTS_BASE_DIR = PROJECT_ROOT / "artifacts"
logger.debug(f"Calculated ARTIFACTS_BASE_DIR: {ARTIFACTS_BASE_DIR}")


def get_artifact_path(*subdirs: str, filename: str = None) -> Path:
    """
    Constructs a path within the artifacts directory. Creates the directory if it doesn't exist.

    Args:
        *subdirs: Variable number of subdirectory names within the artifacts base.
        filename: Optional filename to append to the path.

    Returns:
        The full Path object to the artifact directory or file.
    """
    artifact_dir = ARTIFACTS_BASE_DIR.joinpath(*subdirs)
    artifact_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    if filename:
        return artifact_dir / filename
    else:
        return artifact_dir


if __name__ == "__main__":
    # Example Usage
    log_path = get_artifact_path("logs", filename="test.log")
    print(f"Log path: {log_path}")

    report_dir = get_artifact_path("reports", "run_123")
    print(f"Report directory: {report_dir}")

    # Example of getting just the base logs directory
    logs_base_dir = get_artifact_path("logs")
    print(f"Logs base directory: {logs_base_dir}")
