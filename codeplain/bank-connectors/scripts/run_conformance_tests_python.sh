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

run_discover() {
  START_DIR="$1"
  START_PYTHONPATH="$START_DIR"

  if [ -f "$START_DIR/fixtures_generator.py" ] && [ -d "$START_DIR/tests" ] && [ ! -f "$START_DIR/tests/fixtures_generator.py" ]; then
    printf '%s\n' 'from fixtures_generator import *' > "$START_DIR/tests/fixtures_generator.py"
  fi

  while IFS= read -r dir; do
    touch "$dir/__init__.py"
    case ":$START_PYTHONPATH:" in
      *":$dir:"*) ;;
      *) START_PYTHONPATH="$START_PYTHONPATH:$dir" ;;
    esac
  done <<EOF
$(find "$START_DIR" -type d ! -name "__pycache__" -print)
EOF

  PYTHONPATH="$START_PYTHONPATH:$SOURCE_DIR_ABS${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m unittest discover -s "$START_DIR" -p "*.py"
}

FOUND_SUBPROJECT=0
for child in "$CONFORMANCE_DIR_ABS"/*; do
  if [ -d "$child" ] && [ "$(basename "$child")" != "__pycache__" ] && find "$child" -type f -name "test*.py" | grep -q .; then
    FOUND_SUBPROJECT=1
    run_discover "$child"
  fi
done

if [ "$FOUND_SUBPROJECT" -eq 0 ]; then
  run_discover "$CONFORMANCE_DIR_ABS"
fi
