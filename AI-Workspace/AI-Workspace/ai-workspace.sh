#!/bin/bash
# AI Workspace Shell Wrapper
# Usage: ./ai-workspace.sh [command] [options]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "${SCRIPT_DIR}/bootstrap.py" "$@"
