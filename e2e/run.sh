#!/usr/bin/env bash
# Real end-to-end browser run. Uses a throwaway work folder, so your clips and settings are untouched.
# The network is blocked for the API and the browser, so no platform is ever contacted.
# Needs: scripts/setup.sh, plus Playwright with Chromium (see e2e/README.md).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOME_WORK="${CLIPFORGE_WORK_DIR:-$HOME/clipforge-work}"
QA="${E2E_QA_DIR:-$HOME_WORK/qa}"
. "$ROOT/scripts/use-linux-node.sh"
use_linux_node
export PATH="$HOME_WORK/tools/bin:$PATH"

RUN="$HOME_WORK/tmp/e2e-run"
rm -rf "$RUN"
mkdir -p "$RUN/work/data/fonts" "$RUN/shots"
ln -s "$HOME_WORK/models" "$RUN/work/models"
[ -e "$HOME_WORK/data/fonts/NotoSansCJK-Bold.ttc" ] && ln -s "$HOME_WORK/data/fonts/NotoSansCJK-Bold.ttc" "$RUN/work/data/fonts/"

ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc2=s=640x360:r=30:d=2 -f lavfi -i sine=f=300:d=2 \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p -c:a aac -shortest "$RUN/small.mp4"

export CLIPFORGE_WORK_DIR="$RUN/work" HF_HOME="$HOME_WORK/models/hf" HF_HUB_OFFLINE=1
export HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost
"$HOME_WORK/venv/bin/clipforge-api" > "$RUN/api.log" 2>&1 &
API=$!
(cd "$ROOT/web" && npm run build > "$RUN/build.log" 2>&1 && exec npm run start) > "$RUN/web.log" 2>&1 &
WEB=$!
cleanup() {
  kill "$API" 2>/dev/null
  "$HOME_WORK/venv/bin/python" - <<'PY'
import psutil
for conn in psutil.net_connections("tcp"):
    if conn.laddr and conn.laddr.port == 3000 and conn.status == psutil.CONN_LISTEN and conn.pid:
        psutil.Process(conn.pid).terminate()
PY
  kill "$WEB" 2>/dev/null
}
trap cleanup EXIT
for _ in $(seq 1 120); do
  curl -sf --noproxy '*' -o /dev/null http://127.0.0.1:3000 && curl -sf --noproxy '*' -o /dev/null http://127.0.0.1:8765/api/health && break
  sleep 1
done

LIBS=""
[ -f "$QA/libpath.txt" ] && LIBS="$(cat "$QA/libpath.txt")"
env -u HTTPS_PROXY -u HTTP_PROXY NODE_PATH="$QA/node_modules" LD_LIBRARY_PATH="$LIBS" PLAYWRIGHT_BROWSERS_PATH="$QA/browsers" \
  E2E_SAMPLE="${E2E_SAMPLE:-$HOME_WORK/media/sample_workshop.mp4}" E2E_SMALL_VIDEO="$RUN/small.mp4" E2E_SHOTS="${E2E_SHOTS:-$RUN/shots}" \
  E2E_SEED_SCRIPT="$ROOT/e2e/seed_channel.py" E2E_PYTHON="$HOME_WORK/venv/bin/python" \
  node "$ROOT/e2e/ui.cjs"
CODE=$?
echo "server errors logged: $(grep -cE ' ERROR |Traceback' "$RUN/api.log")"
echo "temp folders left: $(ls "$RUN/work/tmp" 2>/dev/null | grep -cE '^(render|analysis|preview|audio)-')"
exit $CODE
