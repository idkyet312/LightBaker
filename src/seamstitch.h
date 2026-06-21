// seamstitch.h - least-squares lightmap seam stitching.
//
// Ported/adapted from UE5 GPULightmass' LightmapSeamStitching (originally
// Sebastian Sylvan's MIT seam fixup). Instead of trying to make a UV unwrap
// truly seamless, we accept that charts are cut apart in the atlas and FIX the
// seam as a post-bake global optimization:
//
//   1. Find every mesh edge shared by two triangles whose lightmap UVs differ
//      across the shared edge (a "seam edge"): e.g. a sphere's longitude wrap
//      (u=0 vs u=1) or any chart-to-chart cut.
//   2. Build a least-squares system where, sampled along each seam, the
//      bilinearly-interpolated lightmap value on side A equals that on side B,
//      while every touched texel also stays close to its baked value (covered
//      texels pinned hard, gutter texels free to move).
//   3. Solve per color channel with conjugate gradient and write the result
//      back into the atlas.
//
// Run this AFTER denoise and BEFORE dilation.
#pragma once

#include "scene.h"
#include "lightmap.h"

struct SeamStitchSettings {
    float edgeWeight = 6.0f;       // strength of the "edges must agree" term
    float coveredWeight = 1.0f;    // pull covered texels toward their baked value
    float gutterWeight = 0.1f;     // pull non-covered texels (weaker => free to move)
    float cosNormalThresh = 0.5f;  // only stitch edges whose two sides are ~coplanar
    int   iterations = 10000;      // conjugate-gradient cap
    float tolerance = 1e-6f;       // CG residual stop
};

// Stitch all lightmap seams in 'scene' in-place on lm's HDR pixels.
// Returns the number of seam edges found (0 = nothing to do).
int stitchLightmapSeams(const Scene& scene, Lightmap& lm,
                        const SeamStitchSettings& settings = {});
