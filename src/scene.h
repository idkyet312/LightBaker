// scene.h - materials, scene container, and the hardcoded Cornell box.
#pragma once
#include "geometry.h"
#include "bvh.h"
#include <vector>

struct Material {
    Vec3 albedo{ 0.7f, 0.7f, 0.7f };
    Vec3 emission{ 0.0f, 0.0f, 0.0f };
};

enum class ChartKind {
    Quad,
    Sphere
};

// A lightmap chart in the scene. Quad charts store a planar basis; sphere
// charts store a UV unwrap for a tessellated sphere mesh.
struct Quad {
    ChartKind kind = ChartKind::Quad;

    Vec3 origin;     // corner (uv = 0,0)
    Vec3 edgeU;      // full edge along local U
    Vec3 edgeV;      // full edge along local V
    Vec3 corner11;   // corner (uv = 1,1), normally origin + edgeU + edgeV
    Vec3 normal;     // unit
    Vec3 n00, n10, n11, n01; // chart shading normals at the corners
    Vec3 sphereCenter;
    float sphereRadius = 0.0f;
    float sphereSeamAngle = 0.0f;  // longitude rotation (radians) of the UV seam
    // For a SPHERE chart, the latitude band this chart covers, as fractions of
    // the full pole-to-pole sweep: theta = pi * (sphereV0 + v*(sphereV1-sphereV0)).
    // A full-sphere chart is [0,1]; a multi-chart sphere splits this into bands.
    float sphereV0 = 0.0f, sphereV1 = 1.0f;
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

    // Adds a quad (two triangles) and registers it as its own chart. If
    // 'shadingNormal' is given (non-null), the triangles use it for shading
    // instead of the flat cross(edgeU,edgeV) normal.
    int addQuad(const Vec3& origin, const Vec3& edgeU, const Vec3& edgeV,
                int materialId, const Vec3* shadingNormal = nullptr);

    // Tessellate a UV sphere centered at 'center'. The mesh is unwrapped into
    // 'bands' latitude-band lightmap charts (UE GPULightmass / Godot-style
    // multi-chart unwrap): each band is a low-distortion strip, and the seams
    // between bands + the per-band longitude cut are removed afterwards by the
    // least-squares seam stitcher (see seamstitch.cpp). 'seamAngle' rotates the
    // longitude cut around +Y. Higher stacks/slices = smoother silhouette.
    void addSphere(const Vec3& center, float radius, int materialId,
                   int stacks = 48, int slices = 96, float seamAngle = 0.0f,
                   int bands = 6);

    // Collect emissive quads into 'lights' for next-event estimation.
    void collectLights();

    void buildBVH() { bvh.build(&tris); collectLights(); }
};

// Build the classic Cornell box (open front toward +Z camera).
Scene buildCornellBox();
