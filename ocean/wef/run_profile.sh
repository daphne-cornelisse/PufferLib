#!/usr/bin/env bash
# Build and run wef per-thread rollout SPS.  --gprof builds -pg and runs gprof.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SRC=ocean/wef/profile.c
OUT=./wef_profile
OUT_GPROF=./wef_profile_gprof
REPORT=wef_gprof_report.txt

INCS=(-I./raylib-5.5_linux_amd64/include -I./src -I./vendor -I./ocean/wef)
LIBS=(raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt)
CC_FAST="${CC:-clang}"
CC_PROF="${CC_PROF:-gcc}"

GPROF=0
ARGS=()
for a in "$@"; do
    [[ "$a" == "--gprof" ]] && GPROF=1 || ARGS+=("$a")
done

if [[ $GPROF -eq 1 ]]; then
    echo "Building $OUT_GPROF (-pg, serial)..."
    "$CC_PROF" -pg -O1 -g -DNDEBUG -DWEF_PROFILE_NO_OMP \
        "${INCS[@]}" "$SRC" -o "$OUT_GPROF" "${LIBS[@]}" -DPLATFORM_DESKTOP

    rm -f gmon.out
    # Serial rollout sized for a readable flat profile.
    "$OUT_GPROF" --threads 1 --envs-per-thread "${PROFILE_ENVS:-32}" \
        --resets "${PROFILE_RESETS:-5}" --steps "${PROFILE_STEPS:-150}" \
        "${ARGS[@]}"

    [[ -f gmon.out ]] || { echo "error: gmon.out missing" >&2; exit 1; }
    gprof -b -p "$OUT_GPROF" gmon.out > "$REPORT"
    head -n 50 "$REPORT"
    echo
    echo "Full flat profile: $REPORT"
else
    echo "Building $OUT (OpenMP, -O3)..."
    "$CC_FAST" -O3 -DNDEBUG -fopenmp \
        "${INCS[@]}" "$SRC" -o "$OUT" "${LIBS[@]}" -fopenmp -DPLATFORM_DESKTOP
    echo
    "$OUT" "${ARGS[@]}"
fi
