// baker.h - drives per-texel path-traced integration into the lightmap.
#pragma once
#include "scene.h"
#include "lightmap.h"

struct BakeSettings {
    int spp = 256;        // samples per texel
    int bounces = 4;      // indirect bounces
    int dilatePx = 4;     // dilation iterations
    uint64_t seed = 1234; // base RNG seed (deterministic)
    int threads = 0;      // 0 = hardware concurrency
    bool denoise = true;  // run OIDN on the irradiance before dilation
};

// Fill the lightmap's HDR pixels by integrating radiance per covered texel.
void bake(const Scene& scene, Lightmap& lm, const BakeSettings& settings);
