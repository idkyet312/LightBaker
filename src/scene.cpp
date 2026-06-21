#include "scene.h"
#include <utility>   // std::swap

int Scene::addQuad(const Vec3& origin, const Vec3& edgeU, const Vec3& edgeV, int materialId) {
    Quad q;
    q.origin = origin;
    q.edgeU = edgeU;
    q.edgeV = edgeV;
    q.normal = normalize(cross(edgeU, edgeV));
    q.materialId = materialId;
    q.chartId = (int)quads.size();
    quads.push_back(q);

    // Corners.
    Vec3 a = origin;
    Vec3 b = origin + edgeU;
    Vec3 c = origin + edgeU + edgeV;
    Vec3 d = origin + edgeV;

    // Two triangles (a,b,c) and (a,c,d). UVs map the quad to [0,1]^2 in its chart-local space.
    Triangle t0;
    t0.p0 = a; t0.p1 = b; t0.p2 = c;
    t0.uv0 = {0, 0}; t0.uv1 = {1, 0}; t0.uv2 = {1, 1};
    t0.ng = q.normal; t0.materialId = materialId; t0.chartId = q.chartId;

    Triangle t1;
    t1.p0 = a; t1.p1 = c; t1.p2 = d;
    t1.uv0 = {0, 0}; t1.uv1 = {1, 1}; t1.uv2 = {0, 1};
    t1.ng = q.normal; t1.materialId = materialId; t1.chartId = q.chartId;

    tris.push_back(t0);
    tris.push_back(t1);
    return q.chartId;
}

void Scene::collectLights() {
    lights.clear();
    for (const Quad& q : quads) {
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

// Beveled box: the 6 faces are inset by 'b', and the 12 edges get angled
// chamfer strips that catch a highlight - this removes the razor-sharp 90-degree
// corners that read as "CG". Corners keep a tiny triangular gap that is
// invisible at small bevel sizes. Faces are quads; chamfers reuse addQuad so
// they get their own lightmap charts too.
static void addBeveledBox(Scene& s, const Vec3& lo, const Vec3& hi, int mat, float b) {
    float x0=lo.x, y0=lo.y, z0=lo.z, x1=hi.x, y1=hi.y, z1=hi.z;

    // --- 6 inset faces (each shrunk by b on its two in-plane axes) ---
    // +X face at x1, spanning y,z inset
    s.addQuad({x1, y0+b, z0+b}, {0,(y1-y0)-2*b,0}, {0,0,(z1-z0)-2*b}, mat);
    // -X face at x0
    s.addQuad({x0, y0+b, z0+b}, {0,0,(z1-z0)-2*b}, {0,(y1-y0)-2*b,0}, mat);
    // +Y face at y1
    s.addQuad({x0+b, y1, z0+b}, {0,0,(z1-z0)-2*b}, {(x1-x0)-2*b,0,0}, mat);
    // -Y face at y0
    s.addQuad({x0+b, y0, z0+b}, {(x1-x0)-2*b,0,0}, {0,0,(z1-z0)-2*b}, mat);
    // +Z face at z1
    s.addQuad({x0+b, y0+b, z1}, {(x1-x0)-2*b,0,0}, {0,(y1-y0)-2*b,0}, mat);
    // -Z face at z0
    s.addQuad({x0+b, y0+b, z0}, {0,(y1-y0)-2*b,0}, {(x1-x0)-2*b,0,0}, mat);

    // --- 12 edge chamfer strips (45-degree quads bridging adjacent faces) ---
    float ix0=x0+b, ix1=x1-b, iy0=y0+b, iy1=y1-b, iz0=z0+b, iz1=z1-b;
    Vec3 center{(x0+x1)*0.5f,(y0+y1)*0.5f,(z0+z1)*0.5f};
    auto strip = [&](Vec3 a, Vec3 bb, Vec3 c, Vec3 d){
        // Quad a,b,c,d. Ensure the normal points OUTWARD (away from box center)
        // so the baker gathers light from the room side; swap edges if not.
        Vec3 eU = bb - a, eV = d - a;
        Vec3 mid = (a + bb + c + d) * 0.25f;
        if (dot(cross(eU, eV), mid - center) < 0.0f) std::swap(eU, eV);
        s.addQuad(a, eU, eV, mat);
    };
    // 4 chamfers parallel to X (vary the X span, bevel the YZ corner)
    strip({ix0,y0,iz0},{ix1,y0,iz0},{ix1,iy0,z0},{ix0,iy0,z0}); // -Y/-Z
    strip({ix0,iy0,z1},{ix1,iy0,z1},{ix1,y0,iz1},{ix0,y0,iz1}); // -Y/+Z
    strip({ix0,iy1,z0},{ix1,iy1,z0},{ix1,y1,iz0},{ix0,y1,iz0}); // +Y/-Z
    strip({ix0,y1,iz1},{ix1,y1,iz1},{ix1,iy1,z1},{ix0,iy1,z1}); // +Y/+Z
    // 4 chamfers parallel to Y (vary Y, bevel XZ corner)
    strip({x0,iy0,iz0},{x0,iy1,iz0},{ix0,iy1,z0},{ix0,iy0,z0}); // -X/-Z
    strip({ix1,iy0,z0},{ix1,iy1,z0},{x1,iy1,iz0},{x1,iy0,iz0}); // +X/-Z
    strip({ix0,iy0,z1},{ix0,iy1,z1},{x0,iy1,iz1},{x0,iy0,iz1}); // -X/+Z
    strip({x1,iy0,iz1},{x1,iy1,iz1},{ix1,iy1,z1},{ix1,iy0,z1}); // +X/+Z
    // 4 chamfers parallel to Z (vary Z, bevel XY corner)
    strip({ix0,y0,iz0},{ix0,y0,iz1},{x0,iy0,iz1},{x0,iy0,iz0}); // -X/-Y
    strip({x1,iy0,iz0},{x1,iy0,iz1},{ix1,y0,iz1},{ix1,y0,iz0}); // +X/-Y
    strip({x0,iy1,iz0},{x0,iy1,iz1},{ix0,y1,iz1},{ix0,y1,iz0}); // -X/+Y
    strip({ix1,y1,iz0},{ix1,y1,iz1},{x1,iy1,iz1},{x1,iy1,iz0}); // +X/+Y
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

    // Two boxes inside, with small chamfered edges so the corners catch light
    // instead of reading as razor-sharp CG cubes.
    const float bevel = 0.012f;
    addBeveledBox(s, {0.13f, 0.0f, 0.18f}, {0.42f, 0.55f, 0.47f}, white, bevel); // tall box
    addBeveledBox(s, {0.55f, 0.0f, 0.52f}, {0.82f, 0.30f, 0.79f}, white, bevel); // short box

    s.buildBVH();
    return s;
}
