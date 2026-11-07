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
# Pinned to one commit and checked by hash, because this file is loaded and run by the engine.
YUNET="$WORK/models/face/face_detection_yunet_2023mar.onnx"
YUNET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
if [ ! -s "$YUNET" ]; then
  echo "Downloading the face detector..."
  mkdir -p "$WORK/models/face"
  if curl -fsSL -o "$YUNET.part" \
    "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    && echo "$YUNET_SHA256  $YUNET.part" | sha256sum -c --status; then
    mv "$YUNET.part" "$YUNET"
  else
    rm -f "$YUNET.part"
    echo "Could not download it. Reframing still works with the older detector."
  fi
fi

# Exact word timing for captions, about 1.1 GB in all. Pinned and checked by hash, like the face detector.
# Without a model, that language simply keeps Whisper's own timing.
fetch_aligner() {
  local language=$1 repo=$2 revision=$3 file=$4 model_sha=$5 vocab_sha=$6
  local dir="$WORK/models/align/$language" base="https://huggingface.co/$repo/resolve/$revision"
  [ -s "$dir/model.onnx" ] && [ -s "$dir/vocab.json" ] && return 0
  echo "Downloading the word timing model ($language)..."
  mkdir -p "$dir"
  if curl -fL --progress-bar -o "$dir/model.part" "$base/$file" \
    && echo "$model_sha  $dir/model.part" | sha256sum -c --status \
    && curl -fsSL -o "$dir/vocab.part" "$base/vocab.json" \
    && echo "$vocab_sha  $dir/vocab.part" | sha256sum -c --status; then
    mv "$dir/vocab.part" "$dir/vocab.json"
    mv "$dir/model.part" "$dir/model.onnx"
  else
    rm -f "$dir/model.part" "$dir/vocab.part"
    echo "Could not download it. Captions in this language keep Whisper's timing."
  fi
}
fetch_aligner en Xenova/wav2vec2-base-960h a19f851b3d42865797e410752b4c570c871e4825 onnx/model_int8.onnx \
  a4249bb7b7bcbe391be19980922b0523c0907e976a9e5d5aabfadb40fcea5058 \
  4178db26b3c7570f6a47f14ac6a1c7b32950b8c2800fb097287e53776934f1c5
fetch_aligner es XDcobra/wav2vec2-large-xlsr-53-spanish-ONNX ce8c6ded2261a6dd5b902cbbcd46d330f7e92c37 onnx/model_int8.onnx \
  1eb8a34dd7373b48b57c924db5c5a54c3bbf58f5cc8e9cf2631820e01e393244 \
  f73651195659898d08dbeda085d13b249701b83a6731a96213597457cf6cf53f
fetch_aligner hi onnx-community/indicwav2vec-hindi-ONNX abfc14a6a5bc18d8757ec27d2dc03fe4e322b984 onnx/model_int8.onnx \
  ca8e946d2024ee1c7c7a5aa47c6539cdb3486f2aefbd9f866d7ef19958a75d42 \
  9fc45a96d1020e03131a412574d87ec72cc72bf79c2c813679699234b2f509f1
fetch_aligner ar onnx-community/wav2vec2-base-arabic-without-reused-en-token-ids-v1-ONNX 908b2bead4fe5c3ea5d5e31be904129fda068f74 onnx/model.onnx \
  209d73e012442c76b68a1df998e758876cda48c621bea28dae7e88c252d1773c \
  72cc5d4b0400f54f7bf86fb64e82dcfab956cec742a74587e234b6ac61689e30

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
