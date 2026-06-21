#include "baker.h"
#include "pathtracer.h"
#include "denoiser.h"
#include "seamstitch.h"
#include <thread>
#include <atomic>
#include <vector>
#include <cstdio>

void bake(const Scene& scene, Lightmap& lm, const BakeSettings& settings) {
    const int res = lm.res();
    int nthreads = settings.threads > 0 ? settings.threads
                                        : (int)std::max(1u, std::thread::hardware_concurrency());

    std::atomic<int> nextRow{0};
    std::atomic<int> rowsDone{0};

    auto worker = [&]() {
        for (;;) {
            int y = nextRow.fetch_add(1);
            if (y >= res) break;

            for (int x = 0; x < res; ++x) {
                if (!lm.covered(x, y)) continue;
                SurfacePoint sp = lm.texelToSurface(scene, x, y);
                if (!sp.valid) continue;

                // Deterministic per-texel seed: stable across runs and thread counts.
                RNG rng(settings.seed, (uint64_t)y * res + x + 1);
                Vec3 c = bakePoint(scene, sp.pos, sp.normal,
                                   scene.materials[sp.materialId],
                                   rng, settings.spp, settings.bounces);
                lm.at(x, y) = c;
            }

            int done = rowsDone.fetch_add(1) + 1;
            if ((done % 16) == 0 || done == res) {
                std::printf("\r  baking... %d/%d rows (%.0f%%)", done, res,
                            100.0 * done / res);
                std::fflush(stdout);
            }
        }
    };

    std::printf("Baking %dx%d atlas, %d spp, %d bounces, %d threads\n",
                res, res, settings.spp, settings.bounces, nthreads);

    std::vector<std::thread> pool;
    for (int i = 0; i < nthreads; ++i) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    std::printf("\n");

    // Denoise the raw irradiance BEFORE dilation, so the clean result is what
    // gets bled outward into the chart margins.
    if (settings.denoise)
        denoiseLightmap(lm, scene);

    // Least-squares seam stitching (UE GPULightmass-style): equalize the
    // lightmap across every chart-boundary that shares a 3D mesh edge (e.g. the
    // sphere's longitude wrap), so no visible seam remains. Runs after denoise,
    // before dilation, so the corrected values are what gets bled into margins.
    // Periodic-U wrap cleanup for sphere charts: blend the two meridians
    // flanking phi=0 so independent bake/denoise noise across the wrap doesn't
    // read as a vertical line. (The sphere mesh is welded and sampled with a
    // consistent periodic convention, so this is the only remaining seam term.)
    lm.fixSphereSeams(scene);

    // General least-squares seam stitching for any genuine chart-to-chart cuts
    // (band-to-band joins on the multi-chart sphere; longitude wrap per band).
    stitchLightmapSeams(scene, lm);

    if (settings.dilatePx > 0) {
        std::printf("Dilating %d px...\n", settings.dilatePx);
        lm.dilate(settings.dilatePx);
    }
}
