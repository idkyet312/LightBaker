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

From the **x64 Native Tools Command Prompt for VS 2022** (so MSVC + CUDA are on
PATH):

```sh
cd third_party/RTXNTC
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
      -DNTC_WITH_RENDERER=OFF ^
      -DNTC_WITH_TESTS=OFF ^
      -DDONUT_WITH_DLSS=OFF ^
      -DDONUT_WITH_DX12=OFF
cmake --build build --config Release --target ntc-cli
```

The resulting tool lands at:

```
third_party/RTXNTC/build/bin/ntc-cli.exe
```

> If CMake 4 rejects an old submodule's `cmake_minimum_required`, add
> `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the configure line.

Smoke-test it:

```sh
third_party/RTXNTC/build/bin/ntc-cli.exe --listCudaDevices
```

You should see your RTX 4060 listed.

## 4. Use it from the baker workflow

Once `ntc-cli.exe` exists, the demo flow is:

```sh
# 1. Bake the lightmap to PNG as usual
build\Release\baker.exe --res 512 --spp 256 --bounces 4 --out lightmap.png

# 2. Compress the atlas with real NTC (runs on the GPU)
third_party\RTXNTC\build\bin\ntc-cli.exe ^
    lightmap.png --compress --bitsPerPixel 4 ^
    --decompress --saveCompressed lightmap.ntc

# 3. Decompress back to an image to compare (PSNR is printed during compress)
third_party\RTXNTC\build\bin\ntc-cli.exe ^
    lightmap.ntc --decompress --saveImages ntc_out/

# 4. View the round-tripped atlas
build\Release\viewer.exe --lightmap ntc_out\lightmap.png --res 512
```

This will be wrapped behind a baker flag (`--ntc`) and documented in the README
once `ntc-cli` is confirmed building on this machine.

## Notes / gotchas

- NTC is designed to compress **bundles of correlated PBR maps** (albedo, normal,
  roughness, metalness, AO — up to 16 channels). A single 3-channel irradiance
  lightmap is an unusual input, so expect a smaller win than NVIDIA's headline
  numbers. The point here is a working end-to-end NTC pipeline, not optimal ratio.
- The DX12 Cooperative-Vector "inference on sample" path needs a preview Agility
  SDK + developer-preview driver and is explicitly *not for shipping*. We avoid it
  entirely by using `ntc-cli`'s CUDA decompression.
- `--bitsPerPixel` is tunable between 1.0 and 20; lower = smaller file, lower PSNR.
