#!/usr/bin/env bash
# ntc_roundtrip.sh - end-to-end NVIDIA NTC demo for the baked lightmap atlas.
#
#   bake (optional) -> NTC compress (.ntc) -> NTC decompress (PNG) -> report
#
# Requires a built ntc-cli (see NTC_SETUP.md) and CUDA 12.9 on PATH. Run from
# the repo root:  bash scripts/ntc_roundtrip.sh [atlas.png] [bitsPerPixel]
#
# With no atlas argument it bakes a fresh 512x512 atlas first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLI="third_party/RTXNTC/bin/windows-x64/ntc-cli.exe"
CUDA_BIN="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"
[ -d "$CUDA_BIN" ] && export PATH="$CUDA_BIN:$PATH"

if [ ! -x "$CLI" ]; then
    echo "ERROR: ntc-cli not found at $CLI"
    echo "Build it first - see NTC_SETUP.md."
    exit 1
fi

ATLAS="${1:-}"
BPP="${2:-4}"

if [ -z "$ATLAS" ]; then
    ATLAS="ntc_demo.png"
    echo ">> Baking a fresh atlas -> $ATLAS"
    ./build/Release/baker.exe --res 512 --density 110 --spp 64 --bounces 3 --out "$ATLAS" >/dev/null
fi

base="${ATLAS%.*}"
NTC="${base}.ntc"
OUTDIR="${base}_ntc_out"
mkdir -p "$OUTDIR"

echo ">> Compressing $ATLAS -> $NTC  (target ${BPP} bpp)"
"$CLI" "$ATLAS" --compress --bitsPerPixel "$BPP" --decompress --saveCompressed "$NTC" \
    | grep -E "Selected latent|Overall PSNR|File size|Per-texture|GDeflate" || true

echo ">> Decompressing $NTC -> $OUTDIR/"
"$CLI" "$NTC" --decompress --saveImages "$OUTDIR" \
    | grep -E "decompression time|Saved image" || true

# Size report: NTC's real win is vs RAW (uncompressed), not vs PNG (already
# compressed). PNG width/height live in the IHDR chunk as two big-endian
# uint32s at byte offset 16; read them with od (no Python dependency).
read W H < <(od -An -j16 -N8 -tu1 "$ATLAS" | awk '{
    print ($1*16777216 + $2*65536 + $3*256 + $4), ($5*16777216 + $6*65536 + $7*256 + $8)
}')
RAW=$((W * H * 3))
PNG=$(stat -c%s "$ATLAS")
NTCB=$(stat -c%s "$NTC")
echo ""
echo "=================  NTC round-trip summary  ================="
printf "  atlas             : %sx%s, 3 channels\n" "$W" "$H"
printf "  raw (uncompressed): %s bytes\n" "$RAW"
printf "  PNG               : %s bytes\n" "$PNG"
awk -v n="$NTCB" -v raw="$RAW" -v wh="$((W*H))" 'BEGIN{
    printf "  NTC (.ntc)        : %s bytes  (%.2f bpp)\n", n, n*8/wh
    printf "  NTC vs raw        : %.2fx smaller\n", raw/n
}'
echo "  decompressed PNG  : $OUTDIR/$(basename "$ATLAS")"
echo "==========================================================="
