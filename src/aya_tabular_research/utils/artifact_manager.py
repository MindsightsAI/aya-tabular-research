from pathlib import Path

# Define the base directory for artifacts relative to the project root
# Assuming the project root is the parent of the 'src' directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ARTIFACTS_BASE_DIR = PROJECT_ROOT / "artifacts"


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
