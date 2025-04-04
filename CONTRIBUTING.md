# Contributing to AYA Guided Inquiry Framework

First off, thank you for considering contributing! We welcome contributions from the community to help improve and extend the capabilities of this collaborative research framework.

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth contribution process.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Server](#running-the-server)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Workflow](#development-workflow)
  - [Branching](#branching)
  - [Commits](#commits)
- [Project Structure Overview](#project-structure-overview)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. We expect all contributors to follow it. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- **Python:** Version >=3.10, <=3.13 (Check `pyproject.toml` for exact constraints).
- **`uv`:** Required for dependency management and building. Install via `pip install uv` or official instructions.
- **Node.js/npm/npx:** Required for running the MCP Inspector (`npx @modelcontextprotocol/inspector`).
- **Git:** For version control.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd aya-guided-inquiry # Or the actual project directory name
    ```
2.  **Create a virtual environment:** (Recommended)

    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

    # Or using conda
    # conda create -n aya-inquiry python=3.11 # Or desired version
    # conda activate aya-inquiry
    ```

3.  **Sync dependencies:**
    ```bash
    uv sync
    ```
    This installs all necessary dependencies, including development dependencies if configured in `pyproject.toml`, based on `uv.lock`.

### Running the Server (Development)

The recommended way to run the server during development is using the `run-inspector.sh` script:

```bash
./run-inspecotr.sh
```

This script handles syncing dependencies, building the project, and launching the server within the MCP Inspector, providing a convenient way to test MCP interactions.

Refer to `docs/architecture.md` (Section 8) for configuration options (e.g., logging level).

## How to Contribute

### Reporting Bugs

- Check the existing issues to see if the bug has already been reported.
- If not, open a new issue. Provide a clear title and description, steps to reproduce the bug, expected behavior, and actual behavior. Include relevant logs or error messages if possible.

### Suggesting Enhancements

- Open a new issue to discuss your enhancement idea.
- Explain the motivation behind the enhancement and how it would improve the framework.
- Outline your proposed implementation approach if you have one.

### Pull Requests

1.  **Fork the repository** and create your branch from `main` (or the relevant development branch).
2.  **Ensure dependencies are synced** (`uv sync`). This should install development dependencies if configured.
3.  **Make your changes,** adhering to the [Coding Standards](#coding-standards).
4.  **Add tests** for your changes ([Testing](#testing)). Ensure all tests pass.
5.  **Update documentation** if your changes affect architecture, user flow, or configuration ([Documentation](#documentation)).
6.  **Ensure your code lints** without errors.
7.  **Create a pull request** targeting the `main` branch (or as specified by maintainers).
8.  **Write a clear PR description:** Explain the "what" and "why" of your changes. Link to the relevant issue if applicable.
9.  Be prepared to address feedback and make changes during the review process.

## Development Workflow

### Branching

- Create feature branches from `main` (e.g., `feat/improve-planner-strategy`, `fix/report-handler-validation`).
- Use clear branch names.

### Commits

- Write clear, concise commit messages. Follow conventional commit guidelines if adopted by the project (e.g., `feat: Add confidence scoring to Planner`).
- Commit logical units of work.

## Project Structure Overview

Refer to Section 7 ("Proposed Modular Project Structure") in `docs/architecture.md` for a detailed breakdown of the codebase modules and their responsibilities. Understanding this structure is crucial for making targeted contributions. Key areas include:

- `core/`: State management, core data models.
- `planning/`: Strategic decision-making logic.
- `execution/`: Instruction building and report handling.
- `storage/`: Data storage interfaces and implementations (e.g., TabularStore).
- `mcp_interface.py`: MCP primitive handlers.

## Coding Standards

- **Style Guide:** Follow PEP 8 guidelines.
- **Linting:** Use linters like `ruff` or `flake8` (Configuration should be added to `pyproject.toml`). Ensure code passes lint checks before submitting a PR.
- **Typing:** Use Python type hints extensively for clarity and static analysis. Run `mypy` checks if configured.
- **Readability:** Write clear, understandable code with appropriate comments where necessary.

## Testing

- **Unit Tests:** Add unit tests for new functions, classes, or methods, especially within the core logic components (`planning`, `execution`, `storage`, `core`). Place tests in a corresponding `tests/` directory structure.
- **Integration Tests:** (To be developed) Tests covering the interaction between different modules (e.g., `Planner` -> `InstructionBuilder`, `ReportHandler` -> `DataStore`).
- **Framework:** Use `pytest`.
- **Coverage:** Aim for good test coverage for new code.
- Run tests using `pytest` from the project root.

## Documentation

- If your contribution changes the architecture, user flow, configuration, or adds new features/tools/prompts, update the relevant documentation in the `docs/` directory (especially `architecture.md` and potentially `user_flow.md`).
- Add docstrings to new functions, classes, and methods.

Thank you for contributing!
