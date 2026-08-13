#!/usr/bin/env bash
# Build ocean/wef/wef_env*.so for Python import.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="ocean/wef/wef_env${EXT}"

# Prefer headless software raylib (no X11). Source tree has headers + libraylib.a in src/.
RAY_INC=""
RAY_LIB=""
if [ -f "raylib-6.0_memory/src/raylib.h" ] && [ -f "raylib-6.0_memory/src/libraylib.a" ]; then
  RAY_INC="raylib-6.0_memory/src"
  RAY_LIB="raylib-6.0_memory/src"
elif [ -f "raylib-6.0_linux_amd64/include/raylib.h" ]; then
  RAY_INC="raylib-6.0_linux_amd64/include"
  RAY_LIB="raylib-6.0_linux_amd64/lib"
else
  echo "Missing raylib; run ./build.sh wef --headless once to download." >&2
  exit 1
fi

PY_INCLUDES="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or "")')"
if [ -z "$PY_INCLUDES" ] || [ ! -f "$PY_INCLUDES/Python.h" ]; then
  PY_INCLUDES="$(python3-config --includes 2>/dev/null | sed 's/-I//g' | awk '{print $1}')"
fi

INCLUDES=(
  -I"$ROOT"
  -I"$ROOT/src"
  -I"$ROOT/ocean/wef"
  -I"$ROOT/$RAY_INC"
  -I"$PY_INCLUDES"
  -I"$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
  -I"$("$PY" -c 'import numpy; print(numpy.get_include())')"
)

LIBS=(
  -L"$ROOT/$RAY_LIB"
  -Wl,-rpath,"$ROOT/$RAY_LIB"
  "$ROOT/$RAY_LIB/libraylib.a"
  -lm -lpthread -ldl
)

# PLATFORM_MEMORY for headless; SIMD for pufferenv.h
CXX="${CXX:-c++}"
echo "Building $OUT with $CXX ..."
$CXX -O2 -shared -std=c++17 -fPIC -mavx2 -mfma \
  -DPLATFORM_MEMORY -DPLATFORM=PLATFORM_MEMORY \
  -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
  "${INCLUDES[@]}" \
  ocean/wef/binding.cpp \
  -o "$OUT" \
  "${LIBS[@]}"

echo "Built $OUT"
"$PY" -c "import sys; sys.path.insert(0, 'ocean/wef'); import wef_env; print('ok', wef_env.OBS_SIZE, wef_env.NUM_ACTIONS)"
