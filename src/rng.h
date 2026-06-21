// rng.h - PCG32 random generator + cosine-weighted hemisphere sampling.
#pragma once
#include <cstdint>
#include "vecmath.h"

// PCG32 (O'Neill 2014). Tiny, fast, good statistical quality.
struct RNG {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;

    explicit RNG(uint64_t seed = 0, uint64_t seq = 1) {
        state = 0;
        inc = (seq << 1u) | 1u;
        nextU32();
        state += seed;
        nextU32();
    }

    uint32_t nextU32() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
    }

    // float in [0,1)
    float nextFloat() { return (nextU32() >> 8) * (1.0f / 16777216.0f); }
};

constexpr float PI = 3.14159265358979323846f;

// Cosine-weighted hemisphere direction in tangent space (+Z up). pdf = cosTheta / PI.
inline Vec3 cosineSampleHemisphere(float u1, float u2) {
    float r = std::sqrt(u1);
    float phi = 2.0f * PI * u2;
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    return {x, y, z};
}

// Transform a tangent-space direction to world space given a unit normal n.
inline Vec3 tangentToWorld(const Vec3& local, const Vec3& n) {
    Vec3 t, b;
    buildONB(n, t, b);
    return t * local.x + b * local.y + n * local.z;
}
