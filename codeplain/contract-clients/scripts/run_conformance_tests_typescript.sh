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

if [ -f "$CONFORMANCE_DIR_ABS/package.json" ]; then
  cd "$CONFORMANCE_DIR_ABS"
  npm install --ignore-scripts
  npm test
elif [ -f "$SOURCE_DIR_ABS/package.json" ]; then
  cd "$SOURCE_DIR_ABS"
  npm install --ignore-scripts
  if npm run | grep -qE '(^| )test($|:)'; then
    npm test
  fi
else
  find "$CONFORMANCE_DIR_ABS" -type f \( -name '*.ts' -o -name '*.tsx' \) -print | grep -q .
fi
