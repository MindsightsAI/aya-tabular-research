#!/bin/bash
# Script to run aya-tabular-research with proper error handling

set -e  # Exit immediately if a command exits with a non-zero status
cd /home/maxbaluev/aya

rm -f mcp_server.log
rm -f aya_research_data_phase3.parquet
rm -f aya_research_meta_phase3.json

# source ~/.bashrc # Potentially prints to stdout, interfering with MCP
# source ~/.profile # Potentially prints to stdout, interfering with MCP

export CLIENT_PORT=8082
export SERVER_PORT=9000 

uv sync
uv build
#echo "Running aya-tabular-research server..."
npx @modelcontextprotocol/inspector \
  aya-tabular-research
  
  
#echo "Server stopped with status $?" 