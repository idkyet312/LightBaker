// geometry.h - triangle primitive, ray-triangle intersection, AABB.
#pragma once
#include "vecmath.h"
#include <vector>

struct Vec2 { float x = 0, y = 0; };

struct Triangle {
    Vec3 p0, p1, p2;        // world-space positions
    Vec2 uv0, uv1, uv2;     // lightmap UV2 (filled in by the atlas builder)
    Vec3 ng;                // geometric normal (unit)
    Vec3 n0, n1, n2;        // shading normals at the vertices (unit)
    int materialId = 0;
    int chartId = -1;       // which atlas chart this tri belongs to

    Vec3 centroid() const { return (p0 + p1 + p2) * (1.0f / 3.0f); }
};

// A hit record for closest-hit queries.
struct Hit {
    float t = 1e30f;
    float u = 0, v = 0;     // barycentric (w = 1-u-v for p0)
    int tri = -1;
};

struct AABB {
    Vec3 lo{ 1e30f, 1e30f, 1e30f };
    Vec3 hi{ -1e30f, -1e30f, -1e30f };
    void expand(const Vec3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void expand(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    Vec3 center() const { return (lo + hi) * 0.5f; }
    Vec3 extent() const { return hi - lo; }
};

// Moller-Trumbore. Returns true and fills t,u,v on hit within (ray.tmin, ray.tmax).
inline bool intersectTri(const Triangle& tri, const Ray& ray, float& t, float& u, float& v) {
    const Vec3 e1 = tri.p1 - tri.p0;
    const Vec3 e2 = tri.p2 - tri.p0;
    const Vec3 pv = cross(ray.dir, e2);
    const float det = dot(e1, pv);
    if (std::fabs(det) < 1e-12f) return false;
    const float inv = 1.0f / det;
    const Vec3 tv = ray.origin - tri.p0;
    u = dot(tv, pv) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    const Vec3 qv = cross(tv, e1);
    v = dot(ray.dir, qv) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    t = dot(e2, qv) * inv;
    return t > ray.tmin && t < ray.tmax;
}

// Slab test for ray vs AABB given precomputed inverse direction.
inline bool intersectAABB(const AABB& box, const Vec3& o, const Vec3& invD, float tmin, float tmax) {
    for (int a = 0; a < 3; ++a) {
        float t0 = (box.lo[a] - o[a]) * invD[a];
        float t1 = (box.hi[a] - o[a]) * invD[a];
        if (invD[a] < 0.0f) std::swap(t0, t1);
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax < tmin) return false;
    }
    return true;
}
