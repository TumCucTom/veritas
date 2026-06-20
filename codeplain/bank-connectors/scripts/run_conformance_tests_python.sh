#!/usr/bin/env sh
set -eu

SOURCE_DIR="${1:-}"
CONFORMANCE_DIR="${2:-}"
if [ -z "$SOURCE_DIR" ] || [ -z "$CONFORMANCE_DIR" ] || [ ! -d "$SOURCE_DIR" ] || [ ! -d "$CONFORMANCE_DIR" ]; then
  echo "usage: $0 <generated-source-dir> <generated-conformance-dir>" >&2
  exit 2
fi

SOURCE_DIR_ABS=$(cd "$SOURCE_DIR" && pwd -P)
CONFORMANCE_DIR_ABS=$(cd "$CONFORMANCE_DIR" && pwd -P)

touch "$CONFORMANCE_DIR_ABS/__init__.py"
PYTHONPATH="$SOURCE_DIR_ABS${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s "$CONFORMANCE_DIR_ABS" -p "*.py"
