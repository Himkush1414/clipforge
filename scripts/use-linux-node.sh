# Sourced by setup.sh and dev.sh. Under WSL, the first "node" on PATH can be a Windows program,
# which cannot run Linux npm. Find a Linux node and put it first.
is_linux_node() { [ "$("$1" -p process.platform 2>/dev/null | tr -d '\r')" = "linux" ]; }

use_linux_node() {
  local tool dir
  if command -v node >/dev/null && is_linux_node node; then
    return 0
  fi
  for tool in $(type -ap node npm 2>/dev/null); do
    dir="$(dirname "$(readlink -f "$tool")")"
    for candidate in "$(dirname "$tool")/node" "$dir/node" "$dir/../../../bin/node"; do
      if [ -x "$candidate" ] && is_linux_node "$candidate"; then
        PATH="$(cd "$(dirname "$candidate")" && pwd):$PATH"
        export PATH
        return 0
      fi
    done
  done
  echo "No Linux version of Node.js found. Install Node.js inside WSL, not only on Windows." >&2
  exit 1
}
