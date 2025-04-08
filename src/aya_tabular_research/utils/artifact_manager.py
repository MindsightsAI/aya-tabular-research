from pathlib import Path


def find_project_root(
    current_path: Path = None, markers=("pyproject.toml", ".git")
) -> Path:
    """
    Recursively search upwards from the current path to find a directory that contains
    one of the specified marker files or folders.
    """
    if current_path is None:
        current_path = Path(__file__).resolve()

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise FileNotFoundError(f"Could not find project root using markers {markers}")


PROJECT_ROOT = find_project_root()
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
    artifact_dir.mkdir(parents=True, exist_ok=True)

    return artifact_dir / filename if filename else artifact_dir


if __name__ == "__main__":
    log_path = get_artifact_path("logs", filename="test.log")
    print(f"Log path: {log_path}")

    report_dir = get_artifact_path("reports", "run_123")
    print(f"Report directory: {report_dir}")

    logs_base_dir = get_artifact_path("logs")
    print(f"Logs base directory: {logs_base_dir}")
