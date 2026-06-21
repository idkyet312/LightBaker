// math.h - minimal 3D vector math for the lightmap baker.
#pragma once
#include <cmath>
#include <algorithm>

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float s) : x(s), y(s), z(s) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator*(const Vec3& b) const { return {x * b.x, y * b.y, z * b.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    Vec3& operator*=(const Vec3& b) { x *= b.x; y *= b.y; z *= b.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float operator[](int i) const { return (&x)[i]; }
    float& operator[](int i) { return (&x)[i]; }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) { return v / length(v); }
inline Vec3 vmin(const Vec3& a, const Vec3& b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline Vec3 vmax(const Vec3& a, const Vec3& b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
inline float maxComp(const Vec3& v) { return std::max(v.x, std::max(v.y, v.z)); }

struct Ray {
    Vec3 origin;
    Vec3 dir;     // assumed normalized
    float tmin = 1e-4f;
    float tmax = 1e30f;
};

// Branchless orthonormal basis from a unit normal (Duff et al. 2017).
inline void buildONB(const Vec3& n, Vec3& t, Vec3& b) {
    float sign = std::copysign(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    t = Vec3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = Vec3(c, sign + n.y * n.y * a, -n.y);
}
