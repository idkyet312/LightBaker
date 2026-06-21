#include "seamstitch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Minimal sparse matrix with sorted, grow-on-demand rows (CSR-ish). Enough for
// AtA in the seam least-squares: square, symmetric, accessed by (row,col)+=.
// ---------------------------------------------------------------------------
struct SparseMat {
    struct Row {
        std::vector<int> idx;     // sorted column indices
        std::vector<float> val;   // matching coefficients
        float& at(int col) {
            auto it = std::lower_bound(idx.begin(), idx.end(), col);
            size_t p = it - idx.begin();
            if (it == idx.end() || *it != col) {
                idx.insert(it, col);
                val.insert(val.begin() + p, 0.0f);
            }
            return val[p];
        }
    };
    std::vector<Row> rows;
    explicit SparseMat(int n) : rows(n) {}
    float& operator()(int r, int c) { return rows[r].at(c); }

    // out = A * x
    void mul(std::vector<float>& out, const std::vector<float>& x) const {
        const int n = (int)rows.size();
        for (int r = 0; r < n; ++r) {
            float s = 0.0f;
            const Row& row = rows[r];
            for (size_t i = 0; i < row.idx.size(); ++i)
                s += row.val[i] * x[row.idx[i]];
            out[r] = s;
        }
    }
};

float dotv(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Solve A x = b for symmetric positive-(semi)definite A via conjugate gradient.
void conjugateGradient(std::vector<float>& x, const SparseMat& A,
                       const std::vector<float>& b, int maxIter, float tol) {
    const size_t n = x.size();
    std::vector<float> r(n), p(n), Ap(n);
    A.mul(Ap, x);
    for (size_t i = 0; i < n; ++i) r[i] = b[i] - Ap[i];
    p = r;
    float rsq = dotv(r, r);
    if (rsq == 0.0f) return;
    for (int it = 0; it < maxIter; ++it) {
        A.mul(Ap, p);
        float denom = dotv(p, Ap);
        if (denom == 0.0f) break;
        float alpha = rsq / denom;
        for (size_t i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
        float rsqNew = dotv(r, r);
        if (std::fabs(rsqNew - rsq) < tol * n) break;
        float beta = rsqNew / rsq;
        for (size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rsq = rsqNew;
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
struct SVec2 { float u = 0, v = 0; };

// Chart-local UV in [0,1] -> fractional atlas PIXEL coordinate, matching the
// display samplers (render.cpp sampleAtlas / viewer sampleLightmap). Sphere
// charts are periodic in U (texel-centered, period chartW): pixel = chartX +
// u*chartW - 0.5. Flat charts span endpoints: pixel = chartX + u*(chartW-1).
SVec2 chartUVToPixel(const Quad& q, SVec2 uv) {
    bool sphere = (q.kind == ChartKind::Sphere);
    float px = sphere ? (q.chartX + uv.u * q.chartW - 0.5f)
                      : (q.chartX + uv.u * std::max(0, q.chartW - 1));
    float h = (float)std::max(0, q.chartH - 1);
    return { px, q.chartY + uv.v * h };
}

// A seam half-edge: a segment in atlas pixel space belonging to one chart.
struct HalfEdge { SVec2 a, b; int chartId; };
struct SeamEdge { HalfEdge e[2]; };

// Number of sample points to place along a seam edge (~3 per texel of length).
int seamSampleCount(const SeamEdge& s) {
    auto len = [](SVec2 a, SVec2 b) {
        float du = b.u - a.u, dv = b.v - a.v;
        return std::sqrt(du * du + dv * dv);
    };
    float l = std::max(2.0f, std::max(len(s.e[0].a, s.e[0].b), len(s.e[1].a, s.e[1].b)));
    return (int)(l * 3.0f);
}

// Hash key for an undirected/world-position-keyed edge. We key on quantized
// world positions (not vertex indices) so unwelded meshes still match.
struct PosKey {
    int64_t ax, ay, az, bx, by, bz;
    bool operator==(const PosKey& o) const {
        return ax == o.ax && ay == o.ay && az == o.az &&
               bx == o.bx && by == o.by && bz == o.bz;
    }
};
struct PosKeyHash {
    size_t operator()(const PosKey& k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&](int64_t v) { h = (h ^ (size_t)v) * 1099511628211ull; };
        mix(k.ax); mix(k.ay); mix(k.az); mix(k.bx); mix(k.by); mix(k.bz);
        return h;
    }
};
int64_t quant(float f) { return (int64_t)std::llround(f * 4096.0); }
PosKey keyOf(const Vec3& a, const Vec3& b) {
    return { quant(a.x), quant(a.y), quant(a.z),
             quant(b.x), quant(b.y), quant(b.z) };
}

// Per-texel optimization unknown, tracked per (atlas pixel, chart).
struct PixelInfo { int x, y, chartId; bool covered; };

// Maps an atlas pixel to its unknown index(es); a shared pixel used by two
// charts becomes two distinct unknowns (INDEPENDENT_CHART_INTERPOLATION).
struct PixelMap {
    int w, h;
    std::vector<std::vector<std::pair<int,int>>> cells; // (pixelInfoIdx, chartId)
    PixelMap(int w_, int h_) : w(w_), h(h_), cells((size_t)w_ * h_) {}
    int find(int x, int y, int chartId) const {
        for (auto& e : cells[(size_t)y * w + x]) if (e.second == chartId) return e.first;
        return -1;
    }
    void add(int x, int y, int idx, int chartId) {
        cells[(size_t)y * w + x].emplace_back(idx, chartId);
    }
};

int wrap(int x, int n) { x %= n; if (x < 0) x += n; return x; }
int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }

// A chart's atlas rect plus whether its U axis is periodic (sphere longitude).
// Tap coordinates that fall outside the chart are resolved EXACTLY like the
// display samplers (render.cpp sampleAtlas / viewer sampleLightmap): sphere
// charts wrap U modulo chartW and clamp V; flat charts clamp both.
struct ChartRect { int x, y, w, h; bool periodicU; };

// Resolve a (possibly out-of-chart) integer tap to a real atlas pixel.
void resolveTap(const ChartRect& cr, int tx, int ty, int& ox, int& oy) {
    if (cr.periodicU) {
        int lx = wrap(tx - cr.x, cr.w);
        ox = cr.x + lx;
    } else {
        ox = clampi(tx, cr.x, cr.x + cr.w - 1);
    }
    oy = clampi(ty, cr.y, cr.y + cr.h - 1);
}

// Compute the 4 bilinear tap unknown-indices + weights for a fractional sample
// inside one chart, wrapping/clamping taps inside the chart's own rect.
void tapsAndWeights(int chartId, const ChartRect& cr, const PixelMap& pm,
                    SVec2 s, int outIx[4], float outW[4]) {
    int tu = (int)std::floor(s.u);
    int tv = (int)std::floor(s.v);
    int xs[4] = { tu, tu + 1, tu + 1, tu };
    int ys[4] = { tv, tv, tv + 1, tv + 1 };
    for (int i = 0; i < 4; ++i) {
        int x, y; resolveTap(cr, xs[i], ys[i], x, y);
        outIx[i] = pm.find(x, y, chartId);
    }
    float fx = s.u - tu, fy = s.v - tv;
    outW[0] = (1 - fx) * (1 - fy);
    outW[1] = fx * (1 - fy);
    outW[2] = fx * fy;
    outW[3] = (1 - fx) * fy;
}

} // namespace

int stitchLightmapSeams(const Scene& scene, Lightmap& lm,
                        const SeamStitchSettings& settings) {
    const int W = lm.res(), H = lm.res();

    // Per-chart atlas rect + periodicity (sphere charts wrap in U).
    std::vector<ChartRect> crs(scene.quads.size());
    for (size_t i = 0; i < scene.quads.size(); ++i) {
        const Quad& q = scene.quads[i];
        crs[i] = { q.chartX, q.chartY, q.chartW, q.chartH,
                   q.kind == ChartKind::Sphere };
    }

    // --- 1. Find seam edges: shared 3D edges whose atlas UVs differ. ---
    // edgeMap: world-edge -> (atlas-pixel A, atlas-pixel B, normal0, normal1, chartId)
    struct EdgeRec { SVec2 pa, pb; Vec3 n0, n1; int chartId; };
    std::unordered_map<PosKey, EdgeRec, PosKeyHash> edgeMap;
    std::vector<SeamEdge> seams;

    for (const Triangle& t : scene.tris) {
        const Quad& q = scene.quads[t.chartId];
        const Vec3 P[3] = { t.p0, t.p1, t.p2 };
        const Vec3 N[3] = { t.n0, t.n1, t.n2 };
        const SVec2 UV[3] = { { t.uv0.x, t.uv0.y }, { t.uv1.x, t.uv1.y }, { t.uv2.x, t.uv2.y } };
        for (int e = 0; e < 3; ++e) {
            int i0 = e, i1 = (e + 1) % 3;
            SVec2 pa = chartUVToPixel(q, UV[i0]);
            SVec2 pb = chartUVToPixel(q, UV[i1]);
            // Look for the reverse edge already stored.
            auto it = edgeMap.find(keyOf(P[i1], P[i0]));
            if (it == edgeMap.end()) {
                edgeMap[keyOf(P[i0], P[i1])] = { pa, pb, N[i0], N[i1], t.chartId };
            } else {
                const EdgeRec& o = it->second;
                bool coplanar =
                    std::fabs(dot(N[i0], o.n1)) >= settings.cosNormalThresh &&
                    std::fabs(dot(N[i1], o.n0)) >= settings.cosNormalThresh;
                // o stored its edge as (P[i1]->P[i0]) direction, so its pixels
                // (o.pa,o.pb) correspond to verts (i1,i0). To align with our
                // (i0,i1) sampling, this side's matching pixels are (o.pb,o.pa).
                bool uvDiffer = std::fabs(o.pb.u - pa.u) > 1e-3f || std::fabs(o.pb.v - pa.v) > 1e-3f ||
                                std::fabs(o.pa.u - pb.u) > 1e-3f || std::fabs(o.pa.v - pb.v) > 1e-3f;
                if (coplanar && uvDiffer) {
                    SeamEdge s;
                    s.e[0] = { pa, pb, t.chartId };
                    s.e[1] = { o.pb, o.pa, o.chartId };
                    seams.push_back(s);
                }
                edgeMap.erase(it);
            }
        }
    }

    if (seams.empty()) return 0;

    // --- 2. Collect the texels touched by seam-edge bilinear taps. ---
    PixelMap pm(W, H);
    std::vector<PixelInfo> pix;
    for (const SeamEdge& s : seams) {
        int ns = seamSampleCount(s);
        for (int side = 0; side < 2; ++side) {
            const HalfEdge& he = s.e[side];
            const ChartRect& cr = crs[he.chartId];
            SVec2 step{ (he.b.u - he.a.u) / (ns - 1), (he.b.v - he.a.v) / (ns - 1) };
            SVec2 sp = he.a;
            for (int i = 0; i < ns; ++i, sp.u += step.u, sp.v += step.v) {
                int tu = (int)std::floor(sp.u), tv = (int)std::floor(sp.v);
                int xs[4] = { tu, tu + 1, tu + 1, tu };
                int ys[4] = { tv, tv, tv + 1, tv + 1 };
                for (int tap = 0; tap < 4; ++tap) {
                    int x, y; resolveTap(cr, xs[tap], ys[tap], x, y);
                    if (pm.find(x, y, he.chartId) == -1) {
                        bool cov = lm.covered(x, y) && lm.chartAt(x, y) == he.chartId;
                        pix.push_back({ x, y, he.chartId, cov });
                        pm.add(x, y, (int)pix.size() - 1, he.chartId);
                    }
                }
            }
        }
    }

    const int n = (int)pix.size();
    if (n == 0) return (int)seams.size();

    // --- 3. Build AtA for the edge-equality constraints. ---
    // For each seam sample: minimize || bilerp_A - bilerp_B ||^2 * edgeWeight^2.
    SparseMat AtA(n);
    const float ew = settings.edgeWeight;
    for (const SeamEdge& s : seams) {
        int ns = seamSampleCount(s);
        SVec2 stepA{ (s.e[0].b.u - s.e[0].a.u) / (ns - 1), (s.e[0].b.v - s.e[0].a.v) / (ns - 1) };
        SVec2 stepB{ (s.e[1].b.u - s.e[1].a.u) / (ns - 1), (s.e[1].b.v - s.e[1].a.v) / (ns - 1) };
        SVec2 spA = s.e[0].a, spB = s.e[1].a;
        for (int i = 0; i < ns; ++i, spA.u += stepA.u, spA.v += stepA.v, spB.u += stepB.u, spB.v += stepB.v) {
            int ia[4], ib[4]; float wa[4], wb[4];
            tapsAndWeights(s.e[0].chartId, crs[s.e[0].chartId], pm, spA, ia, wa);
            tapsAndWeights(s.e[1].chartId, crs[s.e[1].chartId], pm, spB, ib, wb);
            for (int k = 0; k < 4; ++k) { wa[k] *= ew; wb[k] *= ew; }
            // (a-b)(a-b)^T = a a^T + b b^T - a b^T - b a^T
            for (int i2 = 0; i2 < 4; ++i2)
                for (int j = 0; j < 4; ++j) {
                    if (ia[i2] >= 0 && ia[j] >= 0) AtA(ia[i2], ia[j]) += wa[i2] * wa[j];
                    if (ib[i2] >= 0 && ib[j] >= 0) AtA(ib[i2], ib[j]) += wb[i2] * wb[j];
                    if (ia[i2] >= 0 && ib[j] >= 0) AtA(ia[i2], ib[j]) -= wa[i2] * wb[j];
                    if (ib[i2] >= 0 && ia[j] >= 0) AtA(ib[i2], ia[j]) -= wb[i2] * wa[j];
                }
        }
    }

    // --- 4. Data term: keep each texel near its baked value (diagonal). ---
    std::vector<float> dataW(n);
    for (int i = 0; i < n; ++i) {
        float w = pix[i].covered ? settings.coveredWeight : settings.gutterWeight;
        dataW[i] = w;
        AtA(i, i) += w;
    }

    // --- 5. Solve each color channel and write back. ---
    std::vector<float> b(n), x(n), guess(n);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < n; ++i) {
            // Seed value: covered texels use their baked value; gutter texels
            // are seeded by a small 3x3 same-chart dilation so they start sane.
            float val;
            if (pix[i].covered) {
                val = lm.at(pix[i].x, pix[i].y)[c];
            } else {
                float sum = 0; int cnt = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = wrap(pix[i].x + dx, W), ny = wrap(pix[i].y + dy, H);
                        if (lm.covered(nx, ny) && lm.chartAt(nx, ny) == pix[i].chartId) {
                            sum += lm.at(nx, ny)[c]; ++cnt;
                        }
                    }
                val = cnt ? sum / cnt : 0.0f;
            }
            b[i] = val * dataW[i];
            guess[i] = val;
        }
        x = guess;
        conjugateGradient(x, AtA, b, settings.iterations, settings.tolerance);
        for (int i = 0; i < n; ++i) {
            // Only write back into the pixel's own chart (avoid clobbering a
            // neighbouring chart that shares the same atlas texel).
            if (lm.chartAt(pix[i].x, pix[i].y) == pix[i].chartId)
                lm.at(pix[i].x, pix[i].y)[c] = x[i];
        }
    }

    return (int)seams.size();
}
