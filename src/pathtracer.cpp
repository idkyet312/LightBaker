#include "pathtracer.h"

static Vec3 safeNormalize(const Vec3& v, const Vec3& fallback) {
    float len = length(v);
    return (len > 1e-8f) ? (v / len) : fallback;
}

static Vec3 shadingNormalAt(const Triangle& tri, const Hit& hit) {
    float w0 = 1.0f - hit.u - hit.v;
    return safeNormalize(tri.n0 * w0 + tri.n1 * hit.u + tri.n2 * hit.v,
                         tri.ng);
}

// Direct lighting at a diffuse point via next-event estimation: sample a point
// on each area light and add its (shadow-tested) contribution. Returns the
// Next-event estimation: the albedo-free *direct irradiance* arriving at point
// P with normal N from all area lights (one shadow-tested sample per light).
//   E_direct = sum over lights of Le * cosSurf * cosLight / dist^2 * area
// For reflected radiance, multiply by albedo/pi; as an irradiance contribution
// to a lightmap (which stores irradiance/pi), multiply by 1/pi.
static Vec3 directIrradiance(const Scene& scene, const Vec3& P, const Vec3& N,
                             RNG& rng) {
    Vec3 sum(0);
    for (const LightQuad& lt : scene.lights) {
        // Uniform point on the light quad.
        float u = rng.nextFloat(), v = rng.nextFloat();
        Vec3 S = lt.origin + lt.edgeU * u + lt.edgeV * v;
        Vec3 toL = S - P;
        float dist2 = dot(toL, toL);
        float dist = std::sqrt(dist2);
        Vec3 wi = toL / dist;

        float cosSurf = dot(N, wi);
        float cosLight = dot(lt.normal, -wi);
        if (cosSurf <= 0.0f || cosLight <= 0.0f) continue;  // facing away

        // Shadow ray: is the light visible from P?
        if (scene.bvh.occluded(P + N * 1e-4f, wi, 1e-4f, dist - 1e-3f)) continue;

        // Area-sampling pdf = 1/area, so the 1/pdf factor is *area.
        float g = (cosSurf * cosLight) / dist2 * lt.area;
        sum += lt.emission * g;
    }
    return sum;
}

// Iterative path tracer with next-event estimation. Lambertian BRDF; cosine
// sampling makes the indirect estimator throughput *= albedo each bounce.
//
// All direct lighting (incl. light hit directly) is handled by NEE, so a path
// ray that randomly lands on a light contributes NO emission - this avoids
// double-counting and is what removes the small-light noise. The caller is
// responsible for the direct term at the path's origin surface.
Vec3 radiance(const Scene& scene, Ray ray, RNG& rng, int maxBounces) {
    Vec3 L(0);
    Vec3 throughput(1);

    for (int depth = 0; depth <= maxBounces; ++depth) {
        Hit hit;
        if (!scene.bvh.intersect(ray, hit)) {
            break;  // miss -> black background
        }

        const Triangle& tri = scene.tris[hit.tri];
        const Material& mat = scene.materials[tri.materialId];

        Vec3 P = ray.origin + ray.dir * hit.t;
        Vec3 N = shadingNormalAt(tri, hit);
        if (dot(N, ray.dir) > 0.0f) N = -N;

        // Direct lighting via NEE. Reflected radiance = (albedo/pi) * E_direct.
        L += throughput * mat.albedo * (1.0f / PI) * directIrradiance(scene, P, N, rng);

        if (depth == maxBounces) break;

        // Russian roulette after a couple of bounces to bound path length.
        if (depth >= 3) {
            float p = std::min(0.95f, maxComp(throughput));
            if (rng.nextFloat() >= p) break;
            throughput *= 1.0f / p;
        }

        // Cosine-weighted diffuse bounce for indirect light.
        Vec3 local = cosineSampleHemisphere(rng.nextFloat(), rng.nextFloat());
        Vec3 newDir = normalize(tangentToWorld(local, N));
        throughput *= mat.albedo;

        ray.origin = P + N * 1e-4f;
        ray.dir = newDir;
        ray.tmin = 1e-4f;
        ray.tmax = 1e30f;
    }
    return L;
}

Vec3 bakePoint(const Scene& scene, const Vec3& pos, const Vec3& N,
               const Material& mat, RNG& rng, int spp, int maxBounces) {
    // Surfaces that emit light should bake to their emission directly.
    if (maxComp(mat.emission) > 0.0f) return mat.emission;

    Vec3 origin = pos + N * 1e-4f;

    // The stored value S satisfies display = albedo * S, and for a Lambertian
    // surface S = irradiance/pi = (E_direct + E_indirect)/pi.
    //
    // Direct (E_direct/pi): next-event estimation straight to the light. Low
    // variance even for a small light - this is the big noise win. Averaged
    // over spp so its sampling noise also shrinks.
    Vec3 directSum(0);
    for (int s = 0; s < spp; ++s)
        directSum += directIrradiance(scene, pos, N, rng);
    Vec3 S = directSum * (1.0f / spp) * (1.0f / PI);

    // Indirect (E_indirect/pi): cosine-weighted gather. radiance() carries NO
    // emission (NEE owns all direct light), so this is purely bounced light and
    // there is no double-counting with the direct term above.
    Vec3 indirectSum(0);
    for (int s = 0; s < spp; ++s) {
        // Stratify the 2D sample over an NxN grid (+ jitter) to cut clumping.
        int side = (int)std::sqrt((double)spp);
        if (side < 1) side = 1;
        float u1, u2;
        if (s < side * side) {
            int sx = s % side, sy = s / side;
            u1 = (sx + rng.nextFloat()) / side;
            u2 = (sy + rng.nextFloat()) / side;
        } else {
            u1 = rng.nextFloat(); u2 = rng.nextFloat();
        }
        Vec3 local = cosineSampleHemisphere(u1, u2);
        Vec3 dir = normalize(tangentToWorld(local, N));

        Ray r;
        r.origin = origin;
        r.dir = dir;
        r.tmin = 1e-4f;
        r.tmax = 1e30f;
        indirectSum += radiance(scene, r, rng, maxBounces);
    }
    S += indirectSum * (1.0f / spp);

    return S;
}
