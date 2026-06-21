# C++ Lightmap Baker + Real-time Viewer

A small, self-contained path-traced **lightmap baker** in C++17. It bakes diffuse
global illumination (multiple bounces, color bleeding) for a hardcoded **Cornell box**
into a 2D lightmap atlas, renders an offline preview, and ships an interactive
**OpenGL viewer** you can fly around in to inspect the baked lighting on the surfaces.

Two executables:

- **`baker`** — offline path-traced bake. Core dependency is vendored
  [`stb_image_write`](https://github.com/nothings/stb) in `third_party/`. Optionally
  uses **[Intel Open Image Denoise](https://www.openimagedenoise.org/)** if its prebuilt
  SDK is present in `third_party/oidn/` (auto-detected by CMake).
- **`viewer`** — real-time fly-through that applies the baked lightmap to the geometry.
  Uses **GLFW**, **GLAD**, and **GLM** (provided via [vcpkg](https://github.com/microsoft/vcpkg)).

### Denoising (OIDN)

With OIDN present, the baker path-traces at low spp and **denoises** the result, giving
clean output in a fraction of the time (e.g. ~6 s vs ~40 s for equivalent quality). It
denoises the irradiance using albedo + normal aux buffers, so surface detail stays crisp.
To enable it, drop Intel's prebuilt Windows SDK into `third_party/oidn/` (the layout with
`include/`, `lib/`, `bin/`); CMake prints `denoiser: ENABLED` when found. Without it the
baker still works — just bake at higher `--spp`. Toggle per run with `--denoise` /
`--no-denoise` (on by default).

## What it does

1. Builds a Cornell box in code (`buildCornellBox`) — colored walls, two boxes, a ceiling area light.
2. Packs each quad into a shared lightmap atlas (shelf packing + procedural UV2).
3. For every covered texel, reconstructs the world-space surface point and integrates
   incoming radiance with a Monte Carlo **path tracer** (cosine-weighted hemisphere
   sampling, N diffuse bounces, russian roulette). Multithreaded over atlas rows.
4. Dilates chart edges to avoid seams, tonemaps (Reinhard + gamma), and writes:
   - `lightmap.png` — the baked atlas
   - `preview.png` — a camera view that *samples the baked atlas* (not re-traced)

## Build

Requires CMake and a C++17 compiler (MSVC, GCC, or Clang).

**Baker only** (no extra deps):

```sh
cmake -B build
cmake --build build --config Release
```

**Baker + viewer** — pass the vcpkg toolchain so CMake can find GLFW/GLAD/GLM.
Install them once with `vcpkg install glfw3 glad glm`, then:

```sh
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

CMake prints `viewer: ENABLED` when the deps are found (otherwise just the baker is
built). Executables land in `build/Release/` (MSVC) or `build/` (GCC/Clang).

## Run

```sh
# Windows (MSVC)
build\Release\baker.exe --res 512 --spp 256 --bounces 4 --out lightmap.png --preview preview.png

# A faster, noisier preview
build\Release\baker.exe --res 256 --spp 64 --bounces 3
```

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--res N` | 1024 | Atlas resolution (N x N) |
| `--spp N` | 64 | Samples per texel (low is fine with denoise) |
| `--bounces N` | 6 | Indirect diffuse bounces |
| `--density F` | 220 | Texels per world unit (chart sizing) |
| `--denoise` / `--no-denoise` | on | Denoise the bake with OIDN (if compiled in) |
| `--exposure F` | 1.0 | Tonemap exposure |
| `--out FILE` | lightmap.png | Atlas output |
| `--hdr FILE` | (off) | Also write atlas as `.hdr` |
| `--preview FILE` | preview.png | Preview render output |
| `--pw / --ph N` | 640 | Preview width / height |
| `--seed N` | 1234 | RNG seed (deterministic output) |
| `--threads N` | hardware | Worker thread count |

## What to look for

The Cornell box is the canonical GI test. In `preview.png` you should see:

- The box softly lit by the ceiling light with **soft contact shadows** under the boxes.
- **Color bleeding** — a red tint near the left wall and a green tint near the right wall
  spilling onto the floor and boxes. This only appears with multi-bounce GI and is the
  proof the baker works.
- No black seams at chart boundaries (the dilation pass handles this).

`--bounces 0` shows essentially only the emissive light (everything else dark);
raising `--bounces` fills in indirect light. Higher `--spp` reduces noise.

## Viewer — fly around the scene & bake live

The viewer is a small real-time 3D viewer for the Cornell box. It can **bake the
lightmap on demand from inside the window** (press `B`) and apply it to the objects,
or preload a pre-baked atlas from disk.

Just launch it — no bake required up front:

```sh
build\Release\viewer.exe
```

You start in a plain **shaded** view. Press `B` to path-trace the lightmap (runs on a
background thread so the window stays responsive); when it finishes it's applied to the
geometry automatically. Press `F` to toggle the lightmap on/off.

Tune bake quality with `--spp` / `--bounces`, e.g. a higher-quality bake:

```sh
build\Release\viewer.exe --spp 512 --bounces 5
```

You can also preload an atlas baked by the CLI (must use the **same `--res`/`--density`**
the bake used, since those define the atlas layout):

```sh
build\Release\baker.exe  --res 512 --density 110 --spp 512 --bounces 5 --hdr lightmap.hdr
build\Release\viewer.exe --lightmap lightmap.hdr --res 512 --density 110
```

### Controls

| Input | Action |
|-------|--------|
| `W` `A` `S` `D` | Move forward / left / back / right |
| `Space` / `Ctrl` | Move up / down |
| Mouse | Look around |
| Scroll wheel | Increase / decrease movement speed |
| `B` | **Bake the lightmap** (in-viewer, background thread) |
| `F` | Toggle lightmap on/off (off = neutral shaded view) |
| `G` | Toggle screen-space reflections (glossy) |
| `P` | Toggle post (bloom / vignette / film grain) |
| `T` | Toggle procedural surface texture + roughness |
| `U` | Toggle UV-atlas debug view |
| `R` | Reload the `--lightmap` file from disk |
| `Esc` | Release the mouse; press again to quit |

The viewer renders the scene to an HDR G-buffer, then a composite pass adds
**screen-space reflections** (the glossy floor reflects the boxes/walls), **bloom**
around the bright light, a filmic **ACES** tonemap, **vignette**, and subtle **film
grain** — on top of the baked diffuse GI. Toggle each with the keys above.

### Viewer options

| Flag | Default | Meaning |
|------|---------|---------|
| `--lightmap FILE` | (none) | Preload an HDR atlas to display on start |
| `--res N` | 1024 | Atlas resolution |
| `--density F` | 220 | Texel density |
| `--spp N` | 64 | Samples per texel for the in-viewer bake |
| `--bounces N` | 6 | Indirect bounces for the in-viewer bake |
| `--denoise` / `--no-denoise` | on | Denoise the in-viewer bake with OIDN |
| `--exposure F` | 1.4 | Display exposure |
| `--width / --height N` | 1280 / 800 | Window size |

## Source layout

| File | Responsibility |
|------|----------------|
| `src/vecmath.h` | `Vec3`, `Ray`, orthonormal basis |
| `src/rng.h` | PCG32 RNG + cosine-weighted hemisphere sampling |
| `src/geometry.h` | Triangle, Möller–Trumbore, AABB |
| `src/bvh.*` | BVH (closest-hit + occlusion) |
| `src/scene.*` | Materials, quads, `buildCornellBox()` |
| `src/lightmap.*` | Atlas packing, texel↔surface mapping, dilation |
| `src/pathtracer.*` | `radiance()` GI estimator (NEE), `bakePoint()` |
| `src/baker.*` | Multithreaded per-texel baking |
| `src/denoiser.*` | Optional OIDN denoise of the irradiance atlas |
| `src/render.*` | Preview camera sampling the atlas |
| `src/main.cpp` | Baker CLI: tonemap, PNG/HDR output |
| `src/viewer.cpp` | Real-time OpenGL viewer (fly camera + lightmap) |

## Limitations / scope

Diffuse (Lambertian) only — no specular, textures, or refraction. One shared atlas.
The scene is hardcoded; the `Scene` API is structured so an OBJ/glTF loader could be
added later without touching the baker.
