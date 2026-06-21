#include "bvh.h"
#include <algorithm>

void BVH::build(const std::vector<Triangle>* tris) {
    tris_ = tris;
    const int n = (int)tris->size();
    indices_.resize(n);
    for (int i = 0; i < n; ++i) indices_[i] = i;
    nodes_.clear();
    nodes_.reserve(2 * n);
    if (n > 0) buildNode(0, n);
}

static AABB triBounds(const Triangle& t) {
    AABB b;
    b.expand(t.p0); b.expand(t.p1); b.expand(t.p2);
    return b;
}

int BVH::buildNode(int start, int count) {
    int nodeIdx = (int)nodes_.size();
    nodes_.push_back({});
    AABB box;
    for (int i = 0; i < count; ++i)
        box.expand(triBounds((*tris_)[indices_[start + i]]));

    // Leaf cutoff.
    if (count <= 4) {
        Node leaf;
        leaf.box = box;
        leaf.left = -1;
        leaf.start = start;
        leaf.count = count;
        nodes_[nodeIdx] = leaf;
        return nodeIdx;
    }

    // Split along the longest centroid-bounds axis at the median.
    AABB cbox;
    for (int i = 0; i < count; ++i)
        cbox.expand((*tris_)[indices_[start + i]].centroid());
    Vec3 ext = cbox.extent();
    int axis = 0;
    if (ext.y > ext.x) axis = 1;
    if (ext.z > ext[axis]) axis = 2;

    int mid = start + count / 2;
    std::nth_element(
        indices_.begin() + start,
        indices_.begin() + mid,
        indices_.begin() + start + count,
        [&](int a, int b) {
            return (*tris_)[a].centroid()[axis] < (*tris_)[b].centroid()[axis];
        });

    int left = buildNode(start, mid - start);
    int right = buildNode(mid, start + count - mid);

    Node node;
    node.box = box;
    node.left = left;       // right child is implicitly 'right'
    node.start = right;     // reuse 'start' to store the right child index for inner nodes
    node.count = 0;
    nodes_[nodeIdx] = node;
    return nodeIdx;
}

bool BVH::intersect(const Ray& ray, Hit& out) const {
    if (nodes_.empty()) return false;
    const Vec3 invD{ 1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z };

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    bool hit = false;
    float closest = ray.tmax;

    while (sp > 0) {
        const Node& node = nodes_[stack[--sp]];
        if (!intersectAABB(node.box, ray.origin, invD, ray.tmin, closest)) continue;

        if (node.left < 0) { // leaf
            for (int i = 0; i < node.count; ++i) {
                int ti = indices_[node.start + i];
                float t, u, v;
                Ray r = ray; r.tmax = closest;
                if (intersectTri((*tris_)[ti], r, t, u, v)) {
                    closest = t;
                    out.t = t; out.u = u; out.v = v; out.tri = ti;
                    hit = true;
                }
            }
        } else {
            stack[sp++] = node.left;
            stack[sp++] = node.start; // right child
        }
    }
    return hit;
}

bool BVH::occluded(const Vec3& origin, const Vec3& dir, float tmin, float tmax) const {
    if (nodes_.empty()) return false;
    const Vec3 invD{ 1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z };

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const Node& node = nodes_[stack[--sp]];
        if (!intersectAABB(node.box, origin, invD, tmin, tmax)) continue;

        if (node.left < 0) {
            for (int i = 0; i < node.count; ++i) {
                int ti = indices_[node.start + i];
                Ray r; r.origin = origin; r.dir = dir; r.tmin = tmin; r.tmax = tmax;
                float t, u, v;
                if (intersectTri((*tris_)[ti], r, t, u, v)) return true;
            }
        } else {
            stack[sp++] = node.left;
            stack[sp++] = node.start;
        }
    }
    return false;
}
