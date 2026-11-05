#!/usr/bin/env bash
# Installs everything outside the repo, under CLIPFORGE_WORK_DIR (default ~/clipforge-work).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/use-linux-node.sh
. "$ROOT/scripts/use-linux-node.sh"
use_linux_node
for tool in curl tar npm; do
  command -v "$tool" >/dev/null || { echo "Please install $tool first."; exit 1; }
done
WORK="${CLIPFORGE_WORK_DIR:-$HOME/clipforge-work}"
mkdir -p "$WORK/tools/bin" "$WORK/web" "$WORK/cache"

if ! command -v ffmpeg >/dev/null && [ ! -x "$WORK/tools/bin/ffmpeg" ]; then
  echo "Downloading a static FFmpeg build..."
  URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
  curl -fL --progress-bar -o "$WORK/tools/ffmpeg.tar.xz" "$URL"
  mkdir -p "$WORK/tools/ffmpeg"
  tar -xJf "$WORK/tools/ffmpeg.tar.xz" -C "$WORK/tools/ffmpeg" --strip-components=1
  rm "$WORK/tools/ffmpeg.tar.xz"
  ln -sf ../ffmpeg/bin/ffmpeg "$WORK/tools/bin/ffmpeg"
  ln -sf ../ffmpeg/bin/ffprobe "$WORK/tools/bin/ffprobe"
fi

# Captions for Japanese, Korean and Chinese need this font. OFL licensed, about 20 MB.
FONT="$WORK/data/fonts/NotoSansCJK-Bold.ttc"
if [ ! -s "$FONT" ]; then
  echo "Downloading the CJK caption font..."
  mkdir -p "$WORK/data/fonts"
  curl -fL --progress-bar -o "$FONT" \
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/OTC/NotoSansCJK-Bold.ttc" || {
    rm -f "$FONT"
    echo "Could not download it. Other languages still work, and jobs will warn about missing letters."
  }
fi

# Face detection for vertical reframing. MIT licensed, 232 KB. Falls back to an older detector if missing.
YUNET="$WORK/models/face/face_detection_yunet_2023mar.onnx"
if [ ! -s "$YUNET" ]; then
  echo "Downloading the face detector..."
  mkdir -p "$WORK/models/face"
  curl -fsSL -o "$YUNET" \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" || {
    rm -f "$YUNET"
    echo "Could not download it. Reframing still works with the older detector."
  }
fi

echo "Installing the engine..."
if command -v uv >/dev/null; then
  [ -d "$WORK/venv" ] || uv venv --python 3.12 "$WORK/venv"
  EXTRAS="dev,ml"
  command -v nvidia-smi >/dev/null && EXTRAS="dev,ml,gpu"
  UV_CACHE_DIR="$WORK/cache/uv" uv pip install --python "$WORK/venv/bin/python" -e "$ROOT/engine[$EXTRAS]"
else
  [ -d "$WORK/venv" ] || python3 -m venv "$WORK/venv"
  EXTRAS="dev,ml"
  command -v nvidia-smi >/dev/null && EXTRAS="dev,ml,gpu"
  "$WORK/venv/bin/pip" install -e "$ROOT/engine[$EXTRAS]"
fi

echo "Installing the web app..."
# node_modules lives in the work folder because /mnt drives are slow for many small files.
# It is installed there and then linked, since npm replaces a symlink it finds in place.
cp "$ROOT/web/package.json" "$ROOT/web/package-lock.json" "$WORK/web/"
(cd "$WORK/web" && npm ci --no-audit --no-fund)
rm -rf "$ROOT/web/node_modules"
ln -s "$WORK/web/node_modules" "$ROOT/web/node_modules"
echo "Done. Start clipforge with scripts/dev.sh"
