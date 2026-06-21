#include "scene.h"

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

// Helper: build the 5 outward-or-inward-facing faces of an axis-aligned box.
// 'lo'/'hi' are opposite corners. Normals point outward.
static void addBox(Scene& s, const Vec3& lo, const Vec3& hi, int mat) {
    Vec3 dx{ hi.x - lo.x, 0, 0 };
    Vec3 dy{ 0, hi.y - lo.y, 0 };
    Vec3 dz{ 0, 0, hi.z - lo.z };

    // Every face normal must point OUTWARD from the box (into the room) so the
    // baker gathers light from the room side. normal = cross(edgeU, edgeV).

    // Top (y=hi): normal +Y. cross(+X,+Z) = -Y, so use (dz,dx): cross(+Z,+X)=+Y.
    s.addQuad({lo.x, hi.y, lo.z}, dz, dx, mat);              // +Y
    // Bottom (y=lo): normal -Y. cross(+X,+Z) = -Y.
    s.addQuad(lo, dx, dz, mat);                              // -Y
    // Front (z=hi): normal +Z. cross(+X,+Y) = +Z.
    s.addQuad({lo.x, lo.y, hi.z}, dx, dy, mat);             // +Z
    // Back (z=lo): normal -Z. cross(+Y,+X) = -Z.
    s.addQuad(lo, dy, dx, mat);                              // -Z
    // Left (x=lo): normal -X. cross(+Z,+Y) = -X.
    s.addQuad(lo, dz, dy, mat);                              // -X
    // Right (x=hi): normal +X. cross(+Y,+Z) = +X.
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

    // Two boxes inside (taller one rotated-ish via offset; here axis-aligned for simplicity).
    addBox(s, {0.13f, 0.0f, 0.18f}, {0.42f, 0.55f, 0.47f}, white); // tall box
    addBox(s, {0.55f, 0.0f, 0.52f}, {0.82f, 0.30f, 0.79f}, white); // short box

    s.buildBVH();
    return s;
}
