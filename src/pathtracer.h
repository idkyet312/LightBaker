// pathtracer.h - Monte Carlo radiance estimation with diffuse GI bounces.
#pragma once
#include "scene.h"
#include "rng.h"

// Estimate radiance arriving along 'ray' (toward its origin). Diffuse Lambertian only.
// maxBounces = number of indirect bounces after the primary hit.
Vec3 radiance(const Scene& scene, Ray ray, RNG& rng, int maxBounces);

// Estimate irradiance-derived outgoing radiance for a baked surface point:
// sample the hemisphere around N, gather incoming radiance, return albedo * mean.
// (We bake outgoing radiance directly so the preview can sample it as final color.)
Vec3 bakePoint(const Scene& scene, const Vec3& pos, const Vec3& N,
               const Material& mat, RNG& rng, int spp, int maxBounces);
