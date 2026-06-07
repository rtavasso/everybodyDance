#!/usr/bin/env bash
# everybodyDance live sandbox -- one-shot setup + run on macOS (Intel & Apple
# Silicon) and Linux.
#
#   bash scripts/sandbox_unix.sh                 # default camera
#   bash scripts/sandbox_unix.sh --mirror --calibrate 10
#
# Creates a local venv, installs the real-time stack (pose weights auto-download
# on first run), then launches the webcam sandbox. Re-running skips install if
# the venv already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
ARCH="$(uname -m)"
PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: '$PY' not found. Install Python 3.10-3.12 first." >&2
    exit 1
fi

# Intel Macs: the newest MediaPipe with an x86_64 wheel is 0.10.20, which only
# has cp310/cp311/cp312 builds -- so Python must be 3.10, 3.11 or 3.12 here.
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
    MINOR="$("$PY" -c 'import sys; print(sys.version_info[1])')"
    MAJOR="$("$PY" -c 'import sys; print(sys.version_info[0])')"
    if [ "$MAJOR" != "3" ] || [ "$MINOR" -lt 10 ] || [ "$MINOR" -gt 12 ]; then
        echo "error: Intel Mac detected. MediaPipe's last Intel wheel (0.10.20)" >&2
        echo "       supports Python 3.10-3.12 only; you have $("$PY" -V)." >&2
        echo "       Install Python 3.12 (python.org or 'brew install python@3.12')" >&2
        echo "       then re-run:  PYTHON=python3.12 bash scripts/sandbox_unix.sh" >&2
        exit 1
    fi
    echo "Intel Mac: pinning mediapipe==0.10.20 (Python $("$PY" -V | cut -d' ' -f2))."
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    "$PY" -m venv .venv
    ./.venv/bin/python -m pip install --upgrade pip
    echo "Installing the real-time stack (this can take a few minutes)..."
    ./.venv/bin/python -m pip install -r requirements-realtime.txt
else
    echo "Reusing existing .venv (delete it to reinstall)."
fi

if [ "$OS" = "Darwin" ]; then
    echo
    echo "macOS: if the camera stays black, grant camera access to your terminal in"
    echo "       System Settings > Privacy & Security > Camera, then re-run."
fi

echo
echo "Launching live sandbox. Keys: q quit | r restart song | c re-calibrate | SPACE pause"
echo
exec ./.venv/bin/python -m tools.live "$@"
