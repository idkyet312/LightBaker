#!/usr/bin/env bash
# decode_sphere_material.sh - produce the NTC-compressed PBR material the viewer
# puts on the sphere.
#
#   compress MetalPlates013 bundle -> metalplates.ntc  (real NTC, on the GPU)
#   decompress -> assets/sphere_material/*.png          (the maps the viewer loads)
#
# Requires a built ntc-cli (see NTC_SETUP.md) and CUDA 12.9 on PATH. Run from the
# repo root:  bash scripts/decode_sphere_material.sh [bitsPerPixel]
#
# The decoded maps are git-ignored (regenerable); this script reproduces them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLI="third_party/RTXNTC/bin/windows-x64/ntc-cli.exe"
MAT="third_party/RTXNTC/assets/materials/MetalPlates013"
OUT="assets/sphere_material"
CUDA_BIN="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"
[ -d "$CUDA_BIN" ] && export PATH="$CUDA_BIN:$PATH"

if [ ! -x "$CLI" ]; then
    echo "ERROR: ntc-cli not found at $CLI  (build it first - see NTC_SETUP.md)"
    exit 1
fi
if [ ! -f "$MAT/Manifest.json" ]; then
    echo "ERROR: sample material not found at $MAT"
    echo "It ships with the RTXNTC SDK - see NTC_SETUP.md to clone it."
    exit 1
fi

BPP="${1:-4}"
mkdir -p "$OUT"

echo ">> Compressing $MAT  (10 channels, 2K) -> $OUT/metalplates.ntc  (~${BPP} bpp)"
"$CLI" "$MAT/Manifest.json" --compress --bitsPerPixel "$BPP" \
    --saveCompressed "$OUT/metalplates.ntc" \
    | grep -E "Selected latent|Overall PSNR|File size" || true

# --bcFormat none: emit plain PNGs (BCn/DDS output would need --vk/--dx12, which
# we don't build). The viewer samples these as ordinary GL textures.
echo ">> Decompressing -> $OUT/*.png"
"$CLI" "$OUT/metalplates.ntc" --decompress --bcFormat none --saveImages "$OUT" \
    | grep -E "decompression time|Saved image" || true

echo ""
echo "Decoded maps in $OUT/:"
ls -1 "$OUT"/*.png
echo "Done. Launch the viewer with --pbr to texture the sphere."
