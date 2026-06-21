# NVIDIA RTX Neural Texture Compression (NTC) — setup & build

This project can round-trip its baked lightmap atlas through NVIDIA's
[RTX Neural Texture Compression SDK](https://github.com/NVIDIA-RTX/RTXNTC)
as an end-to-end NTC tech demo: **bake → compress to `.ntc` → decompress →
display**, reporting PSNR and the on-disk size saving.

The SDK is large (~588 MB) and is **built from source against CUDA**, so it is
**not committed** to this repo (`third_party/RTXNTC/` is git-ignored). Follow
the steps below once to set it up locally.

> Hardware on this machine: **RTX 4060 (Ada) + driver 595.79** — fully supported
> for NTC compression. NTC compression *requires* an NVIDIA Turing (RTX 2000) or
> newer GPU.

---

## 1. Prerequisites

Already present on this machine:

- Visual Studio 2022 (Community) — MSVC toolset + Windows SDK
- CMake 4.2.2
- An NVIDIA RTX GPU + recent driver

**Still needed — install this yourself:**

- **CUDA Toolkit 12.9** — https://developer.nvidia.com/cuda-12-9-0-download-archive
  - Pick: Windows → x86_64 → 11 → exe (local).
  - During install, "Visual Studio Integration" must be checked (default).
  - ⚠️ Use **12.9**, not CUDA 13 — NVIDIA notes CUDA 13 is incompatible with the
    590.x developer-preview drivers used for DX12 Cooperative Vectors. (We don't
    use CoopVec in this demo, but 12.9 is the validated, safe choice.)
  - After install, open a **new** terminal and confirm:
    ```sh
    nvcc --version        # should print "release 12.9"
    ```

## 2. Get the SDK with all submodules

The SDK is already present in `third_party/RTXNTC/`. If it was not cloned with
`--recursive`, populate its submodules:

```sh
cd third_party/RTXNTC
git submodule update --init --recursive
```

(As of this writing the submodules are already checked out, including
`external/donut`, `libraries/RTXNTC-Library`, and libdeflate on its `gdeflate`
branch.)

## 3. Build only `ntc-cli` (skip the heavy GPU samples)

We only need the command-line tool (`ntc-cli`) for the demo — it does CUDA-based
compression *and* decompression, so no DX12/Vulkan shader plumbing is required.
Turn off the renderer, explorer, tests, and DLSS to keep the build small and
avoid the DX12 preview-Agility-SDK download.

These are the **verified** flags used on this machine (CMake 4.2.2 + VS 2022 +
CUDA 12.9). Run from a shell where the CUDA toolkit is on PATH and `CUDA_PATH`
points at it:

```sh
cd third_party/RTXNTC
export CUDA_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin:$PATH"

cmake -B build -G "Visual Studio 17 2022" -A x64 \
      -DNTC_WITH_RENDERER=OFF \
      -DNTC_WITH_TESTS=OFF \
      -DDONUT_WITH_DLSS=OFF \
      -DDONUT_WITH_DX12=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCUDAToolkit_ROOT="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9" \
      -DCMAKE_CUDA_COMPILER="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin\\nvcc.exe"

cmake --build build --config Release --target ntc-cli
```

Why each non-obvious flag is needed (learned the hard way):

- `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` — CMake 4 rejects the old
  `cmake_minimum_required` in several submodules (donut/glfw/nvrhi). **In
  PowerShell this value gets mangled by backtick line-continuation** (splits into
  `3` + `.5`); run the configure from Bash, or keep it on one line.
- `-DCMAKE_CUDA_COMPILER=...nvcc.exe` + `CUDA_PATH` — without an explicit
  toolkit path the MSBuild CUDA targets fail with *"The CUDA Toolkit v12.9
  directory '' does not exist"* if CUDA isn't already on the process PATH.
- `-DDONUT_WITH_DX12=OFF` — avoids the preview Agility-SDK / DirectStorage
  downloads. We decompress via CUDA, so DX12 isn't needed.

The first build takes a while (it compiles donut, nvrhi, LibNTC's CUDA kernels,
and ~90 HLSL/Slang shaders). The resulting tool lands at:

```
third_party/RTXNTC/bin/windows-x64/ntc-cli.exe
```

(Note: the SDK overrides the output dir to `bin/windows-x64/`, **not**
`build/bin/`.) A harmless `'pwsh.exe' is not recognized` warning during the
build is fine — ShaderMake falls back to `dxc` directly.

Smoke-test it:

```sh
third_party/RTXNTC/bin/windows-x64/ntc-cli.exe --listCudaDevices
# -> Device 0: NVIDIA GeForce RTX 4060 (compute capability 8.9, 8187 MB VRAM)
```

## 4. Run the end-to-end round-trip

Use the helper script — it bakes (if no atlas given), NTC-compresses, decompresses
on the GPU, and prints a size/quality summary:

```sh
# bake a fresh 512x512 atlas and round-trip it at 4 bpp:
bash scripts/ntc_roundtrip.sh

# or round-trip an existing atlas at a chosen bitrate:
bash scripts/ntc_roundtrip.sh lightmap.png 4
```

It runs the equivalent of:

```sh
CLI=third_party/RTXNTC/bin/windows-x64/ntc-cli.exe
# compress (+ report PSNR), then decompress to an image
"$CLI" lightmap.png --compress --bitsPerPixel 4 --decompress --saveCompressed lightmap.ntc
"$CLI" lightmap.ntc --decompress --saveImages ntc_out/
# view the round-tripped atlas
build/Release/viewer.exe --lightmap ntc_out/lightmap.png --res 512
```

### Verified result (512×512 Cornell-box lightmap, 4 bpp, RTX 4060)

| Metric | Value |
|--------|------:|
| Reconstruction PSNR | **54–56 dB** (visually lossless) |
| `.ntc` file size | ~132 KB @ 4.0 bpp |
| vs **raw** (512×512×3 = 786 KB) | **~6× smaller** |
| GPU decompression time | **0.5–1.0 ms** |

> The size win is best measured against *raw* pixels — a single irradiance PNG is
> already entropy-coded, so `.ntc` (132 KB) vs PNG (162 KB) understates it.

## Notes / gotchas

- NTC is designed to compress **bundles of correlated PBR maps** (albedo, normal,
  roughness, metalness, AO — up to 16 channels). A single 3-channel irradiance
  lightmap is an unusual input, so expect a smaller win than NVIDIA's headline
  numbers. The point here is a working end-to-end NTC pipeline, not optimal ratio.
- The DX12 Cooperative-Vector "inference on sample" path needs a preview Agility
  SDK + developer-preview driver and is explicitly *not for shipping*. We avoid it
  entirely by using `ntc-cli`'s CUDA decompression.
- `--bitsPerPixel` is tunable between 1.0 and 20; lower = smaller file, lower PSNR.
