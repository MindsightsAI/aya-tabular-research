#!/bin/bash
# Script to run aya-tabular-research with proper error handling

set -e  # Exit immediately if a command exits with a non-zero status
PROJECT_DIR="/home/maxbaluev/aya"
cd "$PROJECT_DIR"
echo "Running uv sync..."
uv sync
uv build 

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH" # Prepend src directory
echo "PYTHONPATH is: $PYTHONPATH"
uv --directory "$PROJECT_DIR" run aya-tabular-research
echo "Finished."