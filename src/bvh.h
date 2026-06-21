// bvh.h - binary BVH over triangles with median split. Supports closest-hit and occlusion.
#pragma once
#include "geometry.h"
#include <vector>

class BVH {
public:
    void build(const std::vector<Triangle>* tris);

    // Closest hit. Returns true and fills 'out' on hit.
    bool intersect(const Ray& ray, Hit& out) const;

    // Any-hit visibility test for shadow rays. Returns true if something blocks (tmin,tmax).
    bool occluded(const Vec3& origin, const Vec3& dir, float tmin, float tmax) const;

private:
    struct Node {
        AABB box;
        int left = -1;     // child index, or -1 for leaf
        int start = 0;     // first prim (leaf)
        int count = 0;     // prim count (leaf)
    };

    const std::vector<Triangle>* tris_ = nullptr;
    std::vector<int> indices_;   // permutation of triangle indices
    std::vector<Node> nodes_;

    int buildNode(int start, int count);
};
