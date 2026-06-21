// main.cpp - CLI entry point for the lightmap baker.
#include "scene.h"
#include "lightmap.h"
#include "baker.h"
#include "render.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

// ACES filmic tonemap (Narkowicz fit) for one channel.
static inline float aces(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    float v = (x * (a * x + b)) / (x * (c * x + d) + e);
    return std::min(std::max(v, 0.0f), 1.0f);
}

// Tonemap + gamma 2.2 + dither, HDR Vec3 -> 8-bit RGB.
static std::vector<unsigned char> tonemapLDR(const std::vector<Vec3>& hdr, float exposure) {
    std::vector<unsigned char> out(hdr.size() * 3);
    for (size_t i = 0; i < hdr.size(); ++i) {
        Vec3 c = hdr[i] * exposure;
        for (int k = 0; k < 3; ++k) {
            float v = std::pow(aces(c[k]), 1.0f / 2.2f);
            // Deterministic +-0.5 LSB dither to break 8-bit banding.
            float d = (((i * 3 + k) * 2654435761u & 0xFFFFu) / 65535.0f - 0.5f) / 255.0f;
            v = std::min(std::max(v + d, 0.0f), 1.0f);
            out[i * 3 + k] = (unsigned char)(v * 255.0f + 0.5f);
        }
    }
    return out;
}

static void usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "  --res N         atlas resolution (default 512)\n"
        "  --spp N         samples per texel (default 256)\n"
        "  --bounces N     indirect bounces (default 4)\n"
        "  --density F     texels per world unit (default 256)\n"
        "  --exposure F    tonemap exposure (default 1.0)\n"
        "  --out FILE      lightmap atlas PNG (default lightmap.png)\n"
        "  --hdr FILE      also write atlas as .hdr (optional)\n"
        "  --preview FILE  render a preview PNG sampling the bake (optional)\n"
        "  --pw N --ph N   preview width/height (default 640x640)\n"
        "  --seed N        RNG seed (default 1234)\n"
        "  --threads N     worker threads (default = hardware)\n",
        exe);
}

int main(int argc, char** argv) {
    int res = 2048;
    int density = 360;
    BakeSettings bs;
    bs.spp = 64;          // low spp is fine: OIDN denoises the result
    bs.bounces = 6;
    float exposure = 1.0f;
    std::string outPng = "lightmap.png";
    std::string hdrPath, previewPath = "preview.png";
    int pw = 640, ph = 640;

    auto needArg = [&](int& i) -> const char* {
        if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--res") res = std::atoi(needArg(i));
        else if (a == "--spp") bs.spp = std::atoi(needArg(i));
        else if (a == "--bounces") bs.bounces = std::atoi(needArg(i));
        else if (a == "--density") density = std::atoi(needArg(i));
        else if (a == "--exposure") exposure = (float)std::atof(needArg(i));
        else if (a == "--out") outPng = needArg(i);
        else if (a == "--hdr") hdrPath = needArg(i);
        else if (a == "--preview") previewPath = needArg(i);
        else if (a == "--pw") pw = std::atoi(needArg(i));
        else if (a == "--ph") ph = std::atoi(needArg(i));
        else if (a == "--seed") bs.seed = (uint64_t)std::strtoull(needArg(i), nullptr, 10);
        else if (a == "--threads") bs.threads = std::atoi(needArg(i));
        else if (a == "--denoise") bs.denoise = true;
        else if (a == "--no-denoise") bs.denoise = false;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    std::printf("=== C++ Lightmap Baker ===\n");
    Scene scene = buildCornellBox();
    std::printf("Scene: %zu quads, %zu triangles, %zu materials\n",
                scene.quads.size(), scene.tris.size(), scene.materials.size());

    Lightmap lm;
    if (!lm.build(scene, res, (float)density, 2)) {
        std::printf("ERROR: charts overflow the %dx%d atlas. "
                    "Increase --res or lower --density.\n", res, res);
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    bake(scene, lm, bs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("Bake finished in %.2f s\n", secs);

    // Displayable atlas = irradiance * albedo (so dark-albedo walls stay visible).
    // The lightmap itself stores irradiance; albedo is applied here for output.
    std::vector<Vec3> display = lm.displayAtlas(scene, bs.dilatePx);

    // Write atlas PNG (displayable, tonemapped).
    {
        auto ldr = tonemapLDR(display, exposure);
        if (stbi_write_png(outPng.c_str(), res, res, 3, ldr.data(), res * 3))
            std::printf("Wrote atlas: %s (%dx%d)\n", outPng.c_str(), res, res);
        else
            std::printf("ERROR writing %s\n", outPng.c_str());
    }

    // Optional HDR atlas. This stores raw *irradiance* (no albedo) so the
    // real-time viewer can apply per-surface albedo itself.
    if (!hdrPath.empty()) {
        std::vector<float> f(lm.pixels().size() * 3);
        for (size_t i = 0; i < lm.pixels().size(); ++i) {
            f[i * 3 + 0] = lm.pixels()[i].x;
            f[i * 3 + 1] = lm.pixels()[i].y;
            f[i * 3 + 2] = lm.pixels()[i].z;
        }
        if (stbi_write_hdr(hdrPath.c_str(), res, res, 3, f.data()))
            std::printf("Wrote HDR atlas: %s\n", hdrPath.c_str());
    }

    // Optional preview.
    if (!previewPath.empty()) {
        std::printf("Rendering preview %dx%d...\n", pw, ph);
        std::vector<Vec3> prev = renderPreview(scene, lm, pw, ph, 2);
        auto ldr = tonemapLDR(prev, exposure);
        if (stbi_write_png(previewPath.c_str(), pw, ph, 3, ldr.data(), pw * 3))
            std::printf("Wrote preview: %s (%dx%d)\n", previewPath.c_str(), pw, ph);
    }

    std::printf("Done.\n");
    return 0;
}
