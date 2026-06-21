#include "lightmap.h"
#include <algorithm>
#include <cmath>

bool Lightmap::build(Scene& scene, int res, float texelsPerUnit, int padding) {
    res_ = res;
    pixels_.assign((size_t)res * res, Vec3(0));
    coverage_.assign((size_t)res * res, 0);
    chartId_.assign((size_t)res * res, -1);

    // 1. Size each quad's chart in texels from its world extent.
    struct ChartReq { int quad; int w, h; };
    std::vector<ChartReq> reqs;
    reqs.reserve(scene.quads.size());
    for (int qi = 0; qi < (int)scene.quads.size(); ++qi) {
        const Quad& q = scene.quads[qi];
        float lenU = length(q.edgeU);
        float lenV = length(q.edgeV);
        int w = std::max(2, (int)std::ceil(lenU * texelsPerUnit));
        int h = std::max(2, (int)std::ceil(lenV * texelsPerUnit));
        reqs.push_back({ qi, w, h });
    }

    // 2. Shelf packing, tallest-first for better fill.
    std::sort(reqs.begin(), reqs.end(),
              [](const ChartReq& a, const ChartReq& b) { return a.h > b.h; });

    int penX = padding, penY = padding, shelfH = 0;
    for (const ChartReq& r : reqs) {
        int w = r.w, h = r.h;
        if (penX + w + padding > res_) { // new shelf
            penX = padding;
            penY += shelfH + padding;
            shelfH = 0;
        }
        if (penY + h + padding > res_) {
            return false; // overflow
        }
        Quad& q = scene.quads[r.quad];
        q.chartX = penX;
        q.chartY = penY;
        q.chartW = w;
        q.chartH = h;

        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int px = penX + x, py = penY + y;
                coverage_[py * res_ + px] = 1;
                chartId_[py * res_ + px] = q.chartId;
            }

        penX += w + padding;
        shelfH = std::max(shelfH, h);
    }
    return true;
}

SurfacePoint Lightmap::texelToSurface(const Scene& scene, int x, int y) const {
    SurfacePoint sp{};
    int cid = chartId_[y * res_ + x];
    if (cid < 0) { sp.valid = false; return sp; }

    const Quad& q = scene.quads[cid];
    // Local (u,v) in [0,1] from texel center within the chart rect.
    float u = (x - q.chartX + 0.5f) / (float)q.chartW;
    float v = (y - q.chartY + 0.5f) / (float)q.chartH;
    u = std::min(std::max(u, 0.0f), 1.0f);
    v = std::min(std::max(v, 0.0f), 1.0f);

    sp.pos = q.origin + q.edgeU * u + q.edgeV * v;
    sp.normal = q.normal;
    sp.materialId = q.materialId;
    sp.valid = true;
    return sp;
}

std::vector<Vec3> Lightmap::albedoAtlas(const Scene& scene) const {
    std::vector<Vec3> alb((size_t)res_ * res_, Vec3(0));
    for (int y = 0; y < res_; ++y) {
        for (int x = 0; x < res_; ++x) {
            int cid = chartId_[y * res_ + x];
            if (cid < 0) continue;
            const Material& m = scene.materials[scene.quads[cid].materialId];
            // Emissive surfaces display their emission directly; for those the
            // lightmap already holds emission, so a unit albedo passes it through.
            alb[y * res_ + x] = (maxComp(m.emission) > 0.0f) ? Vec3(1.0f) : m.albedo;
        }
    }
    return alb;
}

std::vector<Vec3> Lightmap::normalAtlas(const Scene& scene) const {
    std::vector<Vec3> nrm((size_t)res_ * res_, Vec3(0));
    for (int y = 0; y < res_; ++y) {
        for (int x = 0; x < res_; ++x) {
            int cid = chartId_[y * res_ + x];
            if (cid < 0) continue;
            nrm[y * res_ + x] = scene.quads[cid].normal;
        }
    }
    return nrm;
}

// Bleed valid texels of 'buf' outward by 'iterations' rings, given 'coverage'
// and resolution 'res'. Shared by the lightmap and the display/albedo atlases.
static void dilateBuffer(std::vector<Vec3>& buf, std::vector<uint8_t> filled,
                         int res, int iterations) {
    static const int dx[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
    static const int dy[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };
    for (int it = 0; it < iterations; ++it) {
        std::vector<uint8_t> newFilled = filled;
        std::vector<Vec3> newBuf = buf;
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                if (filled[y * res + x]) continue;
                Vec3 sum(0);
                int n = 0;
                for (int k = 0; k < 8; ++k) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || ny < 0 || nx >= res || ny >= res) continue;
                    if (filled[ny * res + nx]) { sum += buf[ny * res + nx]; ++n; }
                }
                if (n > 0) {
                    newBuf[y * res + x] = sum * (1.0f / n);
                    newFilled[y * res + x] = 1;
                }
            }
        }
        filled.swap(newFilled);
        buf.swap(newBuf);
    }
}

void Lightmap::dilate(int iterations) {
    dilateBuffer(pixels_, coverage_, res_, iterations);
}

std::vector<Vec3> Lightmap::displayAtlas(const Scene& scene, int dilateIters) const {
    std::vector<Vec3> alb = albedoAtlas(scene);
    std::vector<Vec3> out((size_t)res_ * res_, Vec3(0));
    for (size_t i = 0; i < out.size(); ++i)
        if (coverage_[i]) out[i] = pixels_[i] * alb[i];   // irradiance * albedo
    dilateBuffer(out, coverage_, res_, dilateIters);
    return out;
}
