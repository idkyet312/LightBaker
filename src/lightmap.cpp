#include "lightmap.h"
#include <algorithm>
#include <cmath>

static Vec3 safeNormalize(const Vec3& v, const Vec3& fallback) {
    float len = length(v);
    return (len > 1e-8f) ? (v / len) : fallback;
}

static void quadSurface(const Quad& q, float u, float v,
                        Vec3& pos, Vec3& normal) {
    if (q.kind == ChartKind::Sphere) {
        constexpr float kPi = 3.14159265358979323846f;
        // Chart-local v maps into this chart's latitude band [sphereV0,sphereV1].
        float gv = q.sphereV0 + v * (q.sphereV1 - q.sphereV0);
        float theta = kPi * gv;
        float phi = 2.0f * kPi * u + q.sphereSeamAngle;
        float st = std::sin(theta);
        normal = safeNormalize({st * std::cos(phi),
                                std::cos(theta),
                                st * std::sin(phi)},
                               Vec3(0, 1, 0));
        pos = q.sphereCenter + normal * q.sphereRadius;
        return;
    }

    Vec3 p00 = q.origin;
    Vec3 p10 = q.origin + q.edgeU;
    Vec3 p01 = q.origin + q.edgeV;

    // Match the triangle split used by Scene::addQuad/addSphere:
    // (0,0)-(1,0)-(1,1) and (0,0)-(1,1)-(0,1).
    if (u >= v) {
        float w0 = 1.0f - u;
        float w1 = u - v;
        float w2 = v;
        pos = p00 * w0 + p10 * w1 + q.corner11 * w2;
        normal = safeNormalize(q.n00 * w0 + q.n10 * w1 + q.n11 * w2,
                               q.normal);
    } else {
        float w0 = 1.0f - v;
        float w1 = u;
        float w2 = v - u;
        pos = p00 * w0 + q.corner11 * w1 + p01 * w2;
        normal = safeNormalize(q.n00 * w0 + q.n11 * w1 + q.n01 * w2,
                               q.normal);
    }
}

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
        float lenU = 0.0f, lenV = 0.0f;
        if (q.kind == ChartKind::Sphere) {
            constexpr float kPi = 3.14159265358979323846f;
            lenU = 2.0f * kPi * q.sphereRadius;
            lenV = kPi * q.sphereRadius;
        } else {
            Vec3 p00 = q.origin;
            Vec3 p10 = q.origin + q.edgeU;
            Vec3 p01 = q.origin + q.edgeV;
            lenU = std::max(length(p10 - p00), length(q.corner11 - p01));
            lenV = std::max(length(p01 - p00), length(q.corner11 - p10));
        }
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

    quadSurface(q, u, v, sp.pos, sp.normal);
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
            SurfacePoint sp = texelToSurface(scene, x, y);
            if (sp.valid) nrm[y * res_ + x] = sp.normal;
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

void Lightmap::fixSphereSeams(const Scene& scene) {
    // A UV sphere's chart is PERIODIC in U: phi wraps 2pi -> 0, so the left
    // chart edge (column x0) and the right chart edge (column x1) are adjacent
    // meridians on the surface, one texel apart - not the same point. The mesh
    // references u=0 == u=1 at the seam, and any baked/dilated mismatch between
    // those two edges shows up as a vertical line down the sphere.
    //
    // Fix: stitch the two edges as if the atlas wrapped in U. Each edge column
    // is replaced by the average of itself and its true wrap-neighbour on the
    // opposite edge (x0 <-> x1), so the step across the seam vanishes. A short
    // feather inward blends the next few columns toward that shared value so the
    // correction itself doesn't introduce a new ridge.
    // With the periodic (texel-centered, period chartW) convention, columns
    // x0=chartX and x1=chartX+chartW-1 are the two meridians flanking the phi=0
    // wrap. They should be near-equal, but independent bake/denoise noise (the
    // two columns sit far apart in the atlas, so OIDN filtered them with
    // different neighbours) leaves a step. We can't trust the edge columns
    // themselves (the very last column often picks up a boundary artefact), so we
    // reconstruct the wrap as a smooth crossing built from the *interior*
    // neighbours x0+1 and x1-1, which are clean:
    //   ... x1-1 | x1  x0 | x0+1 ...   (wrap is between x1 and x0)
    // Set x1 and x0 to a weighted blend that makes that 4-tap run monotone.
    for (const Quad& q : scene.quads) {
        if (q.kind != ChartKind::Sphere || q.chartW < 4 || q.chartH < 1) continue;
        int x0 = q.chartX;
        int x1 = q.chartX + q.chartW - 1;
        for (int y = q.chartY; y < q.chartY + q.chartH; ++y) {
            Vec3 inL = at(x1 - 1, y);   // interior neighbour left  of the wrap
            Vec3 inR = at(x0 + 1, y);   // interior neighbour right of the wrap
            // Place x1 and x0 evenly between the two clean interior samples so
            // the sequence inL -> x1 -> x0 -> inR is a smooth ramp (no step/dip).
            at(x1, y) = inL * (2.0f / 3.0f) + inR * (1.0f / 3.0f);
            at(x0, y) = inL * (1.0f / 3.0f) + inR * (2.0f / 3.0f);
        }
    }
}

std::vector<Vec3> Lightmap::displayAtlas(const Scene& scene, int dilateIters) const {
    std::vector<Vec3> alb = albedoAtlas(scene);
    std::vector<Vec3> out((size_t)res_ * res_, Vec3(0));
    for (size_t i = 0; i < out.size(); ++i)
        if (coverage_[i]) out[i] = pixels_[i] * alb[i];   // irradiance * albedo
    dilateBuffer(out, coverage_, res_, dilateIters);
    return out;
}
