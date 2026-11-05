#!/usr/bin/env bash
# Starts the engine API and the web UI. Press Ctrl+C to stop both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${CLIPFORGE_WORK_DIR:-$HOME/clipforge-work}"
# shellcheck source=scripts/use-linux-node.sh
. "$ROOT/scripts/use-linux-node.sh"
use_linux_node
export PATH="$WORK/tools/bin:$PATH"
export HF_HOME="$WORK/models/hf"

"$WORK/venv/bin/clipforge-api" &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT
cd "$ROOT/web"
echo "clipforge: open http://localhost:3000"
if [ "${1:-}" = "--prod" ]; then
  npm run build
  npm run start
else
  npm run dev
fi
