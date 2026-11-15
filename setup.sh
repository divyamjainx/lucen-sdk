#!/usr/bin/env bash
# setup.sh - install the Lucen SDK runtime environment. Use it on the robot computer (Jetson, Ubuntu 22.04)
# and on your own Linux computer. Safe to run again.
#
#   bash setup.sh              # virtual environment + dependencies + udev rule for the MC02 board (needs sudo)
#   bash setup.sh --no-udev    # simulation only, no hardware access (no sudo needed)
#   bash setup.sh --extra      # also install the packages for video recording and camera teleoperation
#
# macOS: there is no udev; run `bash setup.sh --no-udev`. For the MC02 serial port name see docs/connect.md, section 4.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${LUCEN_VENV:-$HOME/lucen_env}"
UDEV=1; EXTRA=0
for a in "$@"; do
  case "$a" in
    --no-udev) UDEV=0 ;;
    --extra)   EXTRA=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

echo "=== 1. Python environment: $VENV ==="
python3 --version
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$HERE/requirements.txt"
if [ "$EXTRA" = 1 ]; then "$VENV/bin/pip" install -q -r "$HERE/requirements-extra.txt"; fi

if [ "$UDEV" = 1 ]; then
  echo "=== 2. udev rule: the MC02 board becomes /dev/mc02 (needs sudo) ==="
  # Match on the USB product name, not on the ttyACM number (which changes with the plug-in order).
  # ID_MM_DEVICE_IGNORE keeps ModemManager from probing the port for tens of seconds after plug-in.
  RULE='SUBSYSTEM=="tty", ATTRS{product}=="Lucen MC02 v1", SYMLINK+="mc02", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"'
  echo "$RULE" | sudo tee /etc/udev/rules.d/99-lucen.rules >/dev/null
  sudo udevadm control --reload-rules && sudo udevadm trigger
  sudo usermod -aG dialout "$USER"
  echo "  rule written to /etc/udev/rules.d/99-lucen.rules; log out and in again for the dialout group to apply"
fi

echo "=== 3. Check ==="
"$VENV/bin/python" - <<'PY'
import importlib
for m in ("numpy", "yaml", "serial", "can", "onnx", "onnxruntime", "mujoco"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<12} ok  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  {m:<12} MISSING ({e})")
PY
ls -l /dev/mc02 2>/dev/null || echo "  /dev/mc02 not present (board not plugged in, or re-plug it once after installing the rule)"
echo
echo "Done. For each session:  source $VENV/bin/activate && cd $HERE"
echo "Try it without hardware: python tools/sim2sim.py --policy biped_Laika/policies/stand.onnx --duration 5"
