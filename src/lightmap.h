// lightmap.h - the atlas: chart packing, texel<->surface mapping, HDR storage, dilation.
#pragma once
#include "scene.h"
#include <vector>

struct SurfacePoint {
    Vec3 pos;
    Vec3 normal;
    int materialId;
    bool valid;
};

class Lightmap {
public:
    // Allocate charts for every quad and pack them into a 'res x res' atlas.
    // 'texelsPerUnit' controls density; charts get at least 'padding' texels of margin.
    // Returns false if packing overflows the atlas.
    bool build(Scene& scene, int res, float texelsPerUnit, int padding = 2);

    int res() const { return res_; }

    // Is this texel covered by a chart? (filled during packing)
    bool covered(int x, int y) const { return coverage_[y * res_ + x] != 0; }

    // Which chart owns this texel (-1 if none).
    int chartAt(int x, int y) const { return chartId_[y * res_ + x]; }

    // Map an atlas texel center to a world-space surface point.
    SurfacePoint texelToSurface(const Scene& scene, int x, int y) const;

    // HDR pixel accessors.
    Vec3& at(int x, int y) { return pixels_[y * res_ + x]; }
    const Vec3& at(int x, int y) const { return pixels_[y * res_ + x]; }

    // Bleed valid texel colors outward by 'iterations' rings to hide seams.
    void dilate(int iterations);

    // Make horizontally wrapped sphere charts continuous at their u=0/u=1 edge.
    void fixSphereSeams(const Scene& scene);

    const std::vector<Vec3>& pixels() const { return pixels_; }

    // Per-texel surface albedo (same layout as the lightmap). The lightmap
    // stores irradiance; multiply by this to get displayable outgoing color.
    std::vector<Vec3> albedoAtlas(const Scene& scene) const;

    // Per-texel world-space surface normal (same layout). Used as a denoiser
    // auxiliary buffer to guide edge-preserving filtering.
    std::vector<Vec3> normalAtlas(const Scene& scene) const;

    // Direct write access to the HDR pixels (e.g. to swap in a denoised buffer).
    std::vector<Vec3>& pixelsMut() { return pixels_; }

    // Build the displayable atlas = dilated(irradiance * albedo), matching the
    // chart layout. Used by the offline PNG/HDR output so dark-albedo surfaces
    // stay visible and chart seams stay clean.
    std::vector<Vec3> displayAtlas(const Scene& scene, int dilateIters) const;

private:
    int res_ = 0;
    std::vector<Vec3> pixels_;       // HDR result
    std::vector<uint8_t> coverage_;  // 1 if a chart covers this texel
    std::vector<int> chartId_;       // chart index per texel (-1 = none)
};
