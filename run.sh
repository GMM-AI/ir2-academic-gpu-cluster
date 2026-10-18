#!/bin/zsh
set -eu
project_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python3}" "$project_dir/fgcs.py" "$@"
