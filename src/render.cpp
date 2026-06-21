#include "render.h"
#include "rng.h"
#include <thread>
#include <atomic>
#include <algorithm>

// Bilinear sample of the atlas at chart-local (u,v) in [0,1] for a given chart.
static Vec3 sampleAtlas(const Lightmap& lm, const Quad& q, float u, float v) {
    // Map chart-local UV to atlas texel space (texel centers).
    float fx = q.chartX + u * q.chartW - 0.5f;
    float fy = q.chartY + v * q.chartH - 0.5f;
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;

    auto clampX = [&](int x) { return std::min(std::max(x, 0), lm.res() - 1); };
    int x1 = clampX(x0 + 1), y1 = clampX(y0 + 1);
    x0 = clampX(x0); y0 = clampX(y0);

    Vec3 c00 = lm.at(x0, y0), c10 = lm.at(x1, y0);
    Vec3 c01 = lm.at(x0, y1), c11 = lm.at(x1, y1);
    Vec3 a = c00 * (1 - tx) + c10 * tx;
    Vec3 b = c01 * (1 - tx) + c11 * tx;
    return a * (1 - ty) + b * ty;
}

std::vector<Vec3> renderPreview(const Scene& scene, const Lightmap& lm,
                                int w, int h, int aa) {
    std::vector<Vec3> img((size_t)w * h, Vec3(0));

    // Camera: looking down -Z at the open front of the box centered around (0.5,0.5,0.5).
    Vec3 eye{ 0.5f, 0.5f, 2.05f };
    Vec3 target{ 0.5f, 0.5f, 0.5f };
    Vec3 up{ 0, 1, 0 };
    Vec3 fwd = normalize(target - eye);
    Vec3 right = normalize(cross(fwd, up));
    Vec3 camUp = cross(right, fwd);
    float fov = 50.0f * PI / 180.0f;
    float halfH = std::tan(fov * 0.5f);
    float aspect = (float)w / h;
    float halfW = halfH * aspect;

    std::atomic<int> nextRow{0};
    auto worker = [&]() {
        for (;;) {
            int py = nextRow.fetch_add(1);
            if (py >= h) break;
            for (int px = 0; px < w; ++px) {
                Vec3 accum(0);
                for (int sy = 0; sy < aa; ++sy)
                    for (int sx = 0; sx < aa; ++sx) {
                        float jx = (sx + 0.5f) / aa;
                        float jy = (sy + 0.5f) / aa;
                        float ndcX = (2.0f * (px + jx) / w - 1.0f) * halfW;
                        float ndcY = (1.0f - 2.0f * (py + jy) / h) * halfH;
                        Vec3 dir = normalize(fwd + right * ndcX + camUp * ndcY);

                        Ray ray; ray.origin = eye; ray.dir = dir;
                        Hit hit;
                        if (scene.bvh.intersect(ray, hit)) {
                            const Triangle& tri = scene.tris[hit.tri];
                            // Interpolate chart-local UV2 at the hit.
                            float wgt0 = 1.0f - hit.u - hit.v;
                            float u = wgt0 * tri.uv0.x + hit.u * tri.uv1.x + hit.v * tri.uv2.x;
                            float v = wgt0 * tri.uv0.y + hit.u * tri.uv1.y + hit.v * tri.uv2.y;
                            const Quad& q = scene.quads[tri.chartId];
                            // Atlas stores irradiance; apply surface albedo here
                            // (emissive surfaces pass through, they hold emission).
                            const Material& m = scene.materials[tri.materialId];
                            Vec3 alb = (maxComp(m.emission) > 0.0f) ? Vec3(1.0f) : m.albedo;
                            accum += sampleAtlas(lm, q, u, v) * alb;
                        }
                        // miss -> black
                    }
                img[(size_t)py * w + px] = accum * (1.0f / (aa * aa));
            }
        }
    };

    int nthreads = (int)std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (int i = 0; i < nthreads; ++i) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    return img;
}
