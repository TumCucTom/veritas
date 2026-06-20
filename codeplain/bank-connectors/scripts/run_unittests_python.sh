#!/usr/bin/env sh
set -eu

SOURCE_DIR="${1:-}"
if [ -z "$SOURCE_DIR" ] || [ ! -d "$SOURCE_DIR" ]; then
  echo "usage: $0 <generated-source-dir>" >&2
  exit 2
fi

cd "$SOURCE_DIR"
python3 -m unittest discover
