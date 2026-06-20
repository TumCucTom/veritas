#!/usr/bin/env sh
set -eu

SOURCE_DIR="${1:-}"
if [ -z "$SOURCE_DIR" ] || [ ! -d "$SOURCE_DIR" ]; then
  echo "usage: $0 <generated-source-dir>" >&2
  exit 2
fi

cd "$SOURCE_DIR"

if [ -f package.json ]; then
  npm install --ignore-scripts
  if npm run | grep -qE '(^| )test($|:)'; then
    npm test
  elif [ -f tsconfig.json ]; then
    npx tsc --noEmit
  fi
elif [ -f tsconfig.json ]; then
  npx tsc --noEmit
else
  find . -type f \( -name '*.ts' -o -name '*.tsx' \) -print | grep -q .
fi
