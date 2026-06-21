#include "scene.h"
#include <algorithm>
#include <cmath>

constexpr float kPi = 3.14159265358979323846f;

static Vec3 safeNormalize(const Vec3& v, const Vec3& fallback) {
    float len = length(v);
    return (len > 1e-8f) ? (v / len) : fallback;
}

static int addQuadCorners(Scene& s,
                          const Vec3& p00, const Vec3& p10,
                          const Vec3& p11, const Vec3& p01,
                          int materialId,
                          const Vec3& n00, const Vec3& n10,
                          const Vec3& n11, const Vec3& n01) {
    Quad q;
    q.origin = p00;
    q.edgeU = p10 - p00;
    q.edgeV = p01 - p00;
    q.corner11 = p11;
    Vec3 flat = safeNormalize(cross(p10 - p00, p11 - p00) +
                              cross(p11 - p00, p01 - p00),
                              Vec3(0, 1, 0));
    q.normal = safeNormalize(n00 + n10 + n11 + n01, flat);
    q.n00 = safeNormalize(n00, q.normal);
    q.n10 = safeNormalize(n10, q.normal);
    q.n11 = safeNormalize(n11, q.normal);
    q.n01 = safeNormalize(n01, q.normal);
    q.materialId = materialId;
    q.chartId = (int)s.quads.size();
    s.quads.push_back(q);

    // Two triangles. UVs map the quad to [0,1]^2 in its chart-local space.
    Triangle t0;
    t0.p0 = p00; t0.p1 = p10; t0.p2 = p11;
    t0.uv0 = {0, 0}; t0.uv1 = {1, 0}; t0.uv2 = {1, 1};
    t0.ng = safeNormalize(cross(p10 - p00, p11 - p00), q.normal);
    t0.n0 = q.n00; t0.n1 = q.n10; t0.n2 = q.n11;
    t0.materialId = materialId; t0.chartId = q.chartId;

    Triangle t1;
    t1.p0 = p00; t1.p1 = p11; t1.p2 = p01;
    t1.uv0 = {0, 0}; t1.uv1 = {1, 1}; t1.uv2 = {0, 1};
    t1.ng = safeNormalize(cross(p11 - p00, p01 - p00), q.normal);
    t1.n0 = q.n00; t1.n1 = q.n11; t1.n2 = q.n01;
    t1.materialId = materialId; t1.chartId = q.chartId;

    s.tris.push_back(t0);
    s.tris.push_back(t1);
    return q.chartId;
}

static void addTriangle(Scene& s,
                        const Vec3& p0, const Vec3& p1, const Vec3& p2,
                        const Vec2& uv0, const Vec2& uv1, const Vec2& uv2,
                        const Vec3& n0, const Vec3& n1, const Vec3& n2,
                        int materialId, int chartId) {
    Vec3 fallback = safeNormalize(n0 + n1 + n2, Vec3(0, 1, 0));
    Triangle t;
    t.p0 = p0; t.p1 = p1; t.p2 = p2;
    t.uv0 = uv0; t.uv1 = uv1; t.uv2 = uv2;
    t.ng = safeNormalize(cross(p1 - p0, p2 - p0), fallback);
    if (dot(t.ng, fallback) < 0.0f) t.ng = -t.ng;
    t.n0 = safeNormalize(n0, t.ng);
    t.n1 = safeNormalize(n1, t.ng);
    t.n2 = safeNormalize(n2, t.ng);
    t.materialId = materialId;
    t.chartId = chartId;
    s.tris.push_back(t);
}

int Scene::addQuad(const Vec3& origin, const Vec3& edgeU, const Vec3& edgeV,
                   int materialId, const Vec3* shadingNormal) {
    Vec3 flat = safeNormalize(cross(edgeU, edgeV), Vec3(0, 1, 0));
    Vec3 n = shadingNormal ? safeNormalize(*shadingNormal, flat) : flat;
    Vec3 p10 = origin + edgeU;
    Vec3 p01 = origin + edgeV;
    Vec3 p11 = p10 + edgeV;
    return addQuadCorners(*this, origin, p10, p11, p01,
                          materialId, n, n, n, n);
}

void Scene::addSphere(const Vec3& center, float radius, int materialId,
                      int stacks, int slices, float seamAngle, int bands) {
    stacks = std::max(2, stacks);
    slices = std::max(3, slices);
    bands = std::max(1, std::min(bands, stacks));  // can't have more bands than rows

    // True sphere position/normal from global (u in [0,1] longitude, v in [0,1]
    // pole-to-pole). Must match quadSurface()'s sphere mapping in lightmap.cpp.
    auto pointFromUV = [&](float u, float v) {
        float theta = kPi * v;
        float phi = 2.0f * kPi * u + seamAngle;
        float st = std::sin(theta);
        return Vec3(center.x + radius * st * std::cos(phi),
                    center.y + radius * std::cos(theta),
                    center.z + radius * st * std::sin(phi));
    };

    // Multi-chart unwrap: split the pole-to-pole sweep into 'bands' latitude
    // strips, each its own chart. Each band covers stack rows [rowLo,rowHi) and
    // latitude fraction [bandV0,bandV1]. Within a band the chart-local v is
    // remapped to [0,1]; u stays the full-longitude [0,1] (the per-band wrap and
    // the band-to-band joins are stitched later by stitchLightmapSeams()).
    for (int b = 0; b < bands; ++b) {
        int rowLo = (int)((int64_t)b * stacks / bands);
        int rowHi = (int)((int64_t)(b + 1) * stacks / bands);
        if (rowHi <= rowLo) continue;
        float bandV0 = (float)rowLo / (float)stacks;
        float bandV1 = (float)rowHi / (float)stacks;

        Quad chart;
        chart.kind = ChartKind::Sphere;
        chart.origin = center;
        chart.edgeU = {2.0f * kPi * radius, 0.0f, 0.0f};            // full circumference
        chart.edgeV = {0.0f, kPi * radius * (bandV1 - bandV0), 0.0f}; // this band's arc
        chart.corner11 = chart.origin + chart.edgeU + chart.edgeV;
        chart.normal = {0, 1, 0};
        chart.n00 = chart.n10 = chart.n11 = chart.n01 = chart.normal;
        chart.sphereCenter = center;
        chart.sphereRadius = radius;
        chart.sphereSeamAngle = seamAngle;
        chart.sphereV0 = bandV0;
        chart.sphereV1 = bandV1;
        chart.materialId = materialId;
        chart.chartId = (int)quads.size();
        quads.push_back(chart);

        float bandSpan = bandV1 - bandV0;
        for (int i = rowLo; i < rowHi; ++i) {
            float gv0 = (float)i / (float)stacks;          // global latitude frac
            float gv1 = (float)(i + 1) / (float)stacks;
            float lv0 = (gv0 - bandV0) / bandSpan;         // chart-local v in band
            float lv1 = (gv1 - bandV0) / bandSpan;
            for (int j = 0; j < slices; ++j) {
                float u0 = (float)j / (float)slices;
                float u1 = (float)(j + 1) / (float)slices;

                Vec2 uv00{u0, lv0}, uv10{u1, lv0}, uv11{u1, lv1}, uv01{u0, lv1};
                Vec3 p00 = pointFromUV(u0, gv0);
                Vec3 p10 = pointFromUV(u1, gv0);
                Vec3 p11 = pointFromUV(u1, gv1);
                Vec3 p01 = pointFromUV(u0, gv1);
                Vec3 n00 = safeNormalize(p00 - center, Vec3(0, 1, 0));
                Vec3 n10 = safeNormalize(p10 - center, Vec3(0, 1, 0));
                Vec3 n11 = safeNormalize(p11 - center, Vec3(0, -1, 0));
                Vec3 n01 = safeNormalize(p01 - center, Vec3(0, -1, 0));

                if (i > 0) {
                    addTriangle(*this, p00, p10, p11, uv00, uv10, uv11,
                                n00, n10, n11, materialId, chart.chartId);
                }
                if (i + 1 < stacks) {
                    addTriangle(*this, p00, p11, p01, uv00, uv11, uv01,
                                n00, n11, n01, materialId, chart.chartId);
                }
            }
        }
    }
}

void Scene::collectLights() {
    lights.clear();
    for (const Quad& q : quads) {
        if (q.kind != ChartKind::Quad) continue;
        const Material& m = materials[q.materialId];
        if (maxComp(m.emission) <= 0.0f) continue;
        LightQuad lq;
        lq.origin = q.origin;
        lq.edgeU = q.edgeU;
        lq.edgeV = q.edgeV;
        lq.normal = q.normal;
        lq.emission = m.emission;
        lq.area = length(cross(q.edgeU, q.edgeV));
        lights.push_back(lq);
    }
}

// Helper: axis-aligned box, 6 outward-facing faces, normals into the room.
static void addBox(Scene& s, const Vec3& lo, const Vec3& hi, int mat) {
    Vec3 dx{ hi.x - lo.x, 0, 0 };
    Vec3 dy{ 0, hi.y - lo.y, 0 };
    Vec3 dz{ 0, 0, hi.z - lo.z };
    s.addQuad({lo.x, hi.y, lo.z}, dz, dx, mat);              // +Y
    s.addQuad(lo, dx, dz, mat);                              // -Y
    s.addQuad({lo.x, lo.y, hi.z}, dx, dy, mat);             // +Z
    s.addQuad(lo, dy, dx, mat);                              // -Z
    s.addQuad(lo, dz, dy, mat);                              // -X
    s.addQuad({hi.x, lo.y, lo.z}, dy, dz, mat);             // +X
}

Scene buildCornellBox() {
    Scene s;

    // Materials.
    int white = s.addMaterial({ Vec3(0.73f, 0.73f, 0.73f), Vec3(0) });
    int red   = s.addMaterial({ Vec3(0.65f, 0.05f, 0.05f), Vec3(0) });
    int green = s.addMaterial({ Vec3(0.12f, 0.45f, 0.15f), Vec3(0) });
    int light = s.addMaterial({ Vec3(0.0f), Vec3(18.0f, 15.0f, 9.0f) }); // warm area light

    // Box spans roughly [0,1]^3. Camera looks down -Z from the open +Z front.
    const float L = 1.0f;

    // All wall normals must point INTO the room (toward the light), because
    // the baker gathers light over the hemisphere around the quad normal.
    // normal = normalize(cross(edgeU, edgeV)), so edge order is chosen to face in.

    // Floor (y=0): normal +Y (up). cross(+Z,+X) = +Y.
    s.addQuad({0, 0, 0}, {0, 0, L}, {L, 0, 0}, white);
    // Ceiling (y=L): normal -Y (down). cross(+X,+Z) = -Y.
    s.addQuad({0, L, 0}, {L, 0, 0}, {0, 0, L}, white);
    // Back wall (z=0): normal +Z (toward camera/interior). cross(+X,+Y) = +Z.
    s.addQuad({0, 0, 0}, {L, 0, 0}, {0, L, 0}, white);
    // Left wall (red, x=0): normal +X (into room). cross(+Y,+Z) = +X.
    s.addQuad({0, 0, 0}, {0, L, 0}, {0, 0, L}, red);
    // Right wall (green, x=L): normal -X (into room). cross(+Z,+Y) = -X.
    s.addQuad({L, 0, 0}, {0, 0, L}, {0, L, 0}, green);

    // Ceiling area light: a large quad just below the ceiling, facing DOWN (-Y).
    // cross(+X,+Z) = -Y. Made big so the whole room is well lit.
    {
        float x0 = 0.25f, x1 = 0.75f, z0 = 0.25f, z1 = 0.75f, y = L - 0.001f;
        s.addQuad({x0, y, z0}, {x1 - x0, 0, 0}, {0, 0, z1 - z0}, light);
    }

    // One sharp-edged box and one smooth sphere inside. The sphere uses a
    // multi-chart (latitude-band) lightmap unwrap (UE GPULightmass / Godot
    // style); band-to-band and longitude seams are removed by the least-squares
    // stitcher. The seam cut is rotated toward the back-right corner.
    addBox(s, {0.13f, 0.0f, 0.18f}, {0.42f, 0.55f, 0.47f}, white);              // tall box
    s.addSphere({0.68f, 0.162f, 0.64f}, 0.16f, white, 48, 96, -1.5708f, 6);     // smooth sphere

    s.buildBVH();
    return s;
}
