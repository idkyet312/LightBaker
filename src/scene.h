// scene.h - materials, scene container, and the hardcoded Cornell box.
#pragma once
#include "geometry.h"
#include "bvh.h"
#include <vector>

struct Material {
    Vec3 albedo{ 0.7f, 0.7f, 0.7f };
    Vec3 emission{ 0.0f, 0.0f, 0.0f };
};

// A planar quad in the scene. Stores its own basis so the baker can map
// atlas texels back to world-space surface points.
struct Quad {
    Vec3 origin;     // corner (uv = 0,0)
    Vec3 edgeU;      // full edge along local U
    Vec3 edgeV;      // full edge along local V
    Vec3 normal;     // unit
    int materialId = 0;
    int chartId = -1;

    // Chart placement in the atlas, filled by the lightmap builder.
    int chartX = 0, chartY = 0, chartW = 0, chartH = 0;
};

// An emissive quad, cached for direct light sampling (next-event estimation).
struct LightQuad {
    Vec3 origin, edgeU, edgeV, normal;
    Vec3 emission;
    float area;
};

struct Scene {
    std::vector<Material> materials;
    std::vector<Quad> quads;
    std::vector<Triangle> tris;   // tessellated geometry for ray tracing
    std::vector<LightQuad> lights;
    BVH bvh;

    int addMaterial(const Material& m) {
        materials.push_back(m);
        return (int)materials.size() - 1;
    }

    // Adds a quad (two triangles) and registers it as its own chart.
    int addQuad(const Vec3& origin, const Vec3& edgeU, const Vec3& edgeV, int materialId);

    // Collect emissive quads into 'lights' for next-event estimation.
    void collectLights();

    void buildBVH() { bvh.build(&tris); collectLights(); }
};

// Build the classic Cornell box (open front toward +Z camera).
Scene buildCornellBox();
