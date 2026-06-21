// viewer.cpp - real-time OpenGL viewer that renders the Cornell box with the
// baked lightmap applied, and lets you fly around with a free camera.
//
// Controls:
//   W/A/S/D     move forward/left/back/right
//   Space / Ctrl   move up / down
//   Mouse           look around (hold to capture; ESC releases / quits)
//   Scroll          change movement speed
//   R               reload the lightmap atlas from disk
//   F               toggle fullscreen-ish unlit/lit (lightmap on/off)
//
// The viewer reconstructs the exact same scene + atlas layout the baker used,
// so it must be given the same --res / --density that produced the .hdr atlas.

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "scene.h"
#include "lightmap.h"
#include "baker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>
#include <algorithm>

// ---------------------------------------------------------------------------
// Interleaved vertex: world position + atlas UV + normal + albedo + SSR tag +
// chart rect. The chart rect (atlas texels: x,y,w,h) lets the fragment shader
// sample U-periodic charts (the sphere) with manual wrapped bilinear, so the
// longitude seam doesn't show. w<=0 means "plain clamped sampling".
// ---------------------------------------------------------------------------
struct GLVertex {
    float px, py, pz;
    float u, v;
    float nx, ny, nz;
    float ar, ag, ab;
    float ssrTarget;
    float cx, cy, cw, ch;   // chart rect in atlas texels (cw<=0 => no wrap)
};

// Map a triangle's chart-local UV (0..1 within its quad) to atlas UV (0..1
// across the whole atlas), matching how the baker/preview place charts.
static void chartUVtoAtlas(const Quad& q, int res, float lu, float lv,
                           float& au, float& av) {
    float texX = q.chartX + 0.5f + lu * (float)std::max(0, q.chartW - 1);
    float texY = q.chartY + 0.5f + lv * (float)std::max(0, q.chartH - 1);
    au = texX / (float)res;
    av = texY / (float)res;
}

static std::vector<GLVertex> buildVertices(const Scene& scene, int res) {
    std::vector<GLVertex> verts;
    verts.reserve(scene.tris.size() * 3);
    for (const Triangle& t : scene.tris) {
        const Quad& q = scene.quads[t.chartId];
        const Material& m = scene.materials[t.materialId];
        // Atlas stores irradiance; the shader multiplies by this albedo.
        // Emissive surfaces pass the lightmap (their emission) through unmodified.
        Vec3 alb = (maxComp(m.emission) > 0.0f) ? Vec3(1.0f) : m.albedo;
        const Vec3 p[3] = { t.p0, t.p1, t.p2 };
        const Vec2 uv[3] = { t.uv0, t.uv1, t.uv2 };
        const Vec3 n[3] = { t.n0, t.n1, t.n2 };
        bool isSphere = (q.kind == ChartKind::Sphere);
        float ssrTarget = isSphere ? 0.0f : 1.0f;
        for (int i = 0; i < 3; ++i) {
            float su, sv;        // value stored in the UV attribute
            float cw;            // chart width signal (>0 => shader wraps in U)
            if (isSphere) {
                // Hand the shader the CHART-LOCAL uv + chart rect so it can do
                // U-periodic bilinear itself (the atlas can't GL_REPEAT a region).
                su = uv[i].x; sv = uv[i].y;
                cw = (float)q.chartW;
            } else {
                // Flat charts: pre-bake the atlas uv and disable wrapping.
                chartUVtoAtlas(q, res, uv[i].x, uv[i].y, su, sv);
                cw = 0.0f;
            }
            verts.push_back({ p[i].x, p[i].y, p[i].z, su, sv,
                              n[i].x, n[i].y, n[i].z,
                              alb.x, alb.y, alb.z, ssrTarget,
                              (float)q.chartX, (float)q.chartY, cw, (float)q.chartH });
        }
    }
    return verts;
}

// ---------------------------------------------------------------------------
// Shaders.
// ---------------------------------------------------------------------------
static const char* kVert = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec3 aNormal;
layout(location=3) in vec3 aAlbedo;
layout(location=4) in float aSSRTarget;
layout(location=5) in vec4 aChartRect;   // atlas texels x,y,w,h (w<=0 => no wrap)
uniform mat4 uMVP;
out vec2 vUV;
out vec3 vNormal;
out vec3 vWorldPos;
out vec3 vAlbedo;
out float vSSRTarget;
flat out vec4 vChartRect;   // flat: chart rect is per-triangle, must NOT interpolate
void main() {
    vUV = aUV;
    vNormal = aNormal;
    vWorldPos = aPos;
    vAlbedo = aAlbedo;
    vSSRTarget = aSSRTarget;
    vChartRect = aChartRect;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

// Scene fragment shader: outputs LINEAR HDR radiance to MRT0, and view-space
// normal + per-pixel roughness to MRT1 (for screen-space reflections + post).
static const char* kFrag = R"(#version 330 core
in vec2 vUV;
in vec3 vNormal;
in vec3 vWorldPos;
in vec3 vAlbedo;
in float vSSRTarget;
flat in vec4 vChartRect;    // atlas texels x,y,w,h (w<=0 => plain clamped sampling)

layout(location=0) out vec4 oColor;      // linear HDR radiance
layout(location=1) out vec4 oNormalRough; // viewNormal.xyz (0..1), roughness in .a

uniform sampler2D uLightmap;
uniform float uAtlasRes;    // lightmap atlas resolution (texels)
uniform float uExposure;
uniform int uMode;          // 0 = shaded, 1 = lightmap, 2 = UV debug
uniform vec3 uCamPos;
uniform mat3 uNormalToView; // world->view for normals
uniform int uUseTexture;    // procedural surface texture/roughness on/off
uniform vec3 uLightPos;     // ceiling light center (for a subtle specular)
uniform int uSpec;          // subtle specular highlight on/off

// NTC-decoded PBR material, applied to the sphere only (vSSRTarget < 0.5).
uniform int uUsePBR;            // master toggle
uniform float uPBRTiling;       // material UV repeats across the sphere chart
uniform sampler2D uPBRAlbedo;   // sRGB color
uniform sampler2D uPBRNormal;   // tangent-space normal (DX convention)
uniform sampler2D uPBRRough;    // .r roughness
uniform sampler2D uPBRMetal;    // .r metalness
uniform sampler2D uPBRAO;       // .r ambient occlusion

// Perturb geometric normal N by a tangent-space normal map sample, building the
// tangent frame analytically from screen-space derivatives (no precomputed
// tangents needed). Standard "cotangent frame" trick (Mikkelsen).
vec3 applyNormalMap(vec3 N, vec3 worldPos, vec2 uv, vec3 nTS) {
    vec3 dp1 = dFdx(worldPos), dp2 = dFdy(worldPos);
    vec2 du1 = dFdx(uv),       du2 = dFdy(uv);
    vec3 dp2p = cross(dp2, N), dp1p = cross(N, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    mat3 TBN = mat3(T * inv, B * inv, N);
    return normalize(TBN * nTS);
}

// Sample the lightmap for one fragment. Flat charts use ordinary clamped
// bilinear (chart.w<=0). The sphere chart is PERIODIC in U: phi wraps 2pi->0,
// so its left/right atlas edges are adjacent meridians. GL_CLAMP_TO_EDGE would
// show that wrap as a vertical seam, so we do bilinear by hand here, wrapping
// the column index modulo chartW and clamping the row - exactly matching the
// offline preview's sampler.
vec3 sampleLightmap() {
    if (vChartRect.z <= 0.0) return texture(uLightmap, vUV).rgb;

    float cx = vChartRect.x, cy = vChartRect.y;
    float cw = vChartRect.z, ch = vChartRect.w;
    // Sphere chart: U is PERIODIC with period cw (matches the bake, which samples
    // texel x at u=(x+0.5)/cw). u spans all cw columns and u=1 wraps to u=0, so
    // the wrap blends true neighbour meridians instead of duplicating the seam
    // column. V stays endpoint-clamped over (ch-1). Mirrors render.cpp sampleAtlas.
    float fx = cx + clamp(vUV.x, 0.0, 1.0) * cw - 0.5;
    float fy = cy + 0.5 + clamp(vUV.y, 0.0, 1.0) * max(0.0, ch - 1.0);
    float x0 = floor(fx), y0 = floor(fy - 0.5);
    float tx = fx - x0, ty = (fy - 0.5) - y0;

    // Fetch a single atlas texel with U wrapped within the chart, V clamped.
    // (texelFetch keeps it exact - no reliance on the texture's wrap mode.)
    int W = int(cw);
    vec3 c00 = texelFetch(uLightmap, ivec2(int(cx) + ((int(x0)   - int(cx)) % W + W) % W, clamp(int(y0),   int(cy), int(cy+ch)-1)), 0).rgb;
    vec3 c10 = texelFetch(uLightmap, ivec2(int(cx) + ((int(x0)+1 - int(cx)) % W + W) % W, clamp(int(y0),   int(cy), int(cy+ch)-1)), 0).rgb;
    vec3 c01 = texelFetch(uLightmap, ivec2(int(cx) + ((int(x0)   - int(cx)) % W + W) % W, clamp(int(y0)+1, int(cy), int(cy+ch)-1)), 0).rgb;
    vec3 c11 = texelFetch(uLightmap, ivec2(int(cx) + ((int(x0)+1 - int(cx)) % W + W) % W, clamp(int(y0)+1, int(cy), int(cy+ch)-1)), 0).rgb;
    vec3 a = mix(c00, c10, tx);
    vec3 b = mix(c01, c11, tx);
    return mix(a, b, ty);
}

// Value noise for subtle procedural surface variation.
float hash13(vec3 p){ p=fract(p*0.1031); p+=dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z); }
float vnoise(vec3 x){
    vec3 i=floor(x), f=fract(x); f=f*f*(3.0-2.0*f);
    float n=mix(mix(mix(hash13(i+vec3(0,0,0)),hash13(i+vec3(1,0,0)),f.x),
                    mix(hash13(i+vec3(0,1,0)),hash13(i+vec3(1,1,0)),f.x),f.y),
                mix(mix(hash13(i+vec3(0,0,1)),hash13(i+vec3(1,0,1)),f.x),
                    mix(hash13(i+vec3(0,1,1)),hash13(i+vec3(1,1,1)),f.x),f.y),f.z);
    return n;
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 albedo = vAlbedo;

    // Everything is matte by default (rough = 1 => no reflections). Only the
    // FLOOR (upward normal) is glossy, so walls never reflect. The .a channel
    // is the reflectivity signal the SSR pass keys on.
    float rough = 1.0;
    bool isFloor = (N.y > 0.85);
    if (isFloor) rough = 0.18;          // glossy, but not mirror-polished

    // NTC-decoded PBR material on the sphere (vSSRTarget<0.5 tags the sphere).
    // Reuse the sphere's chart-local UV (phi,theta) as material UV, tiled.
    float metallic = 0.0;
    float ao = 1.0;
    bool isSphere = (vSSRTarget < 0.5);
    if (uUsePBR == 1 && isSphere) {
        vec2 muv = vUV * uPBRTiling;
        albedo = texture(uPBRAlbedo, muv).rgb;
        rough  = clamp(texture(uPBRRough, muv).r, 0.04, 1.0);
        metallic = texture(uPBRMetal, muv).r;
        ao = texture(uPBRAO, muv).r;
        vec3 nTS = texture(uPBRNormal, muv).xyz * 2.0 - 1.0;
        nTS.y = -nTS.y;                  // DX-style normal map -> GL (flip green)
        N = applyNormalMap(N, vWorldPos, muv, nTS);
    }

    // Optional faint surface texture (albedo variation only - does NOT make
    // walls reflective, which was the source of the blotchy artifacts).
    else if (uUseTexture == 1) {
        float n = vnoise(vWorldPos * 18.0) * 0.6 + vnoise(vWorldPos * 60.0) * 0.4;
        albedo *= (0.95 + 0.05 * n);
    }

    vec3 col;
    if (uMode == 1) {
        vec3 irr = sampleLightmap();
        // Metals have (almost) no diffuse; AO darkens crevices. The baked
        // irradiance is the incoming light; albedo/metallic/ao shape outgoing.
        vec3 diffuseCol = albedo * (1.0 - metallic) * ao;
        col = irr * diffuseCol * uExposure;          // linear HDR (no tonemap here)

        // Subtle specular sheen from the ceiling light on ALL surfaces, so
        // nothing reads as dead-flat matte. Broad lobe, Fresnel-boosted at
        // grazing angles, tighter/stronger on the glossy floor. For metals the
        // specular is tinted by the albedo (F0 = albedo) and much stronger.
        if (uSpec == 1) {
            vec3 V = normalize(uCamPos - vWorldPos);
            vec3 Lp = normalize(uLightPos - vWorldPos);
            vec3 H = normalize(V + Lp);
            float ndl = max(dot(N, Lp), 0.0);
            float ndh = max(dot(N, H), 0.0);
            float shin = mix(24.0, 220.0, 1.0 - rough);   // floor = tighter highlight
            float ndv = max(dot(N, V), 0.0);
            vec3 F0 = mix(vec3(0.04), albedo, metallic);
            vec3 fres = F0 + (1.0 - F0) * pow(1.0 - ndv, 5.0);
            vec3 spec = pow(ndh, shin) * ndl * fres;
            float strength = isFloor ? 1.2 : (metallic > 0.5 ? 1.6 : 0.35);
            // Metals also pick up the baked irradiance as a tinted reflection.
            if (metallic > 0.0) spec += F0 * irr * metallic * (1.0 - rough);
            col += spec * strength * uExposure;
        }
    } else if (uMode == 2) {
        col = vec3(vUV, 0.0);
        rough = 1.0;
    } else {
        vec3 L = normalize(uCamPos - vWorldPos);
        float diff = abs(dot(N, L));
        col = albedo * (1.0 - metallic) * ao * (0.25 + 0.75 * diff) * uExposure;
        if (uUsePBR == 1 && isSphere) {
            // A headlight specular so the metal/normal detail is visible without a bake.
            vec3 V = normalize(uCamPos - vWorldPos);
            vec3 H = normalize(V + L);
            vec3 F0 = mix(vec3(0.04), albedo, metallic);
            float shin = mix(16.0, 200.0, 1.0 - rough);
            col += F0 * pow(max(dot(N, H), 0.0), shin) * diff * 1.5 * uExposure;
        }
    }

    oColor = vec4(col, 1.0);
    vec3 vN = normalize(uNormalToView * N);
    // Alpha doubles as a compact SSR tag: floor < 0.3 can emit reflections;
    // 0.5 marks curved objects that should not be SSR targets; 1.0 is stable matte geometry.
    float ssrTag = (vSSRTarget < 0.5) ? 0.5 : rough;
    oNormalRough = vec4(vN * 0.5 + 0.5, ssrTag);
}
)";

// Fullscreen-triangle vertex shader for the post/composite pass.
static const char* kPostVert = R"(#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

// Post/composite: screen-space reflections + bloom + ACES + vignette + grain.
static const char* kPostFrag = R"(#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uScene;       // linear HDR scene color
uniform sampler2D uNormalRough; // view normal + roughness
uniform sampler2D uDepth;       // depth buffer
uniform vec2  uRes;
uniform float uExposure;
uniform mat4  uProj;
uniform mat4  uInvProj;
uniform int   uSSR;             // reflections on/off
uniform int   uPost;            // bloom/vignette/grain on/off
uniform float uTime;

vec3 aces(vec3 x){
    const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0);
}
float hash12(vec2 p){ vec3 p3=fract(vec3(p.xyx)*0.1031); p3+=dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }

// Reconstruct view-space position from depth.
vec3 viewPos(vec2 uv){
    float d = texture(uDepth, uv).r * 2.0 - 1.0;
    vec4 c = vec4(uv * 2.0 - 1.0, d, 1.0);
    vec4 v = uInvProj * c;
    return v.xyz / v.w;
}

float contactFade(vec2 uv, float sourceZ){
    vec2 px = 1.0 / uRes;
    float closer = 0.0;
    for (int i=0;i<8;i++){
        float a = float(i) * 0.785398163;
        vec2 q = uv + vec2(cos(a), sin(a)) * px * 8.0;
        if (q.x<0.0||q.x>1.0||q.y<0.0||q.y>1.0) continue;
        float z = viewPos(q).z;
        closer = max(closer, smoothstep(0.02, 0.14, z - sourceZ));
    }
    return 1.0 - closer * 0.9;
}

vec3 sampleReflection(vec2 uv){
    vec2 px = 1.75 / uRes;
    vec3 c = texture(uScene, uv).rgb * 0.40;
    c += texture(uScene, uv + vec2( px.x, 0.0)).rgb * 0.15;
    c += texture(uScene, uv + vec2(-px.x, 0.0)).rgb * 0.15;
    c += texture(uScene, uv + vec2(0.0,  px.y)).rgb * 0.15;
    c += texture(uScene, uv + vec2(0.0, -px.y)).rgb * 0.15;
    return c;
}

// Screen-space reflection ray-march; returns reflected color and a hit mask.
// Rejects reflective-floor self hits, which otherwise smear the floor back into
// itself as comb-like streaks around contact shadows.
vec3 ssr(vec3 P, vec3 N, out float hitMask){
    hitMask = 0.0;
    vec3 V = normalize(P);             // view dir (camera at origin in view space)
    vec3 R = reflect(V, N);
    if (R.z > 0.0) return vec3(0.0);   // reflecting toward camera plane: skip
    float stepLen = 0.035;
    vec3 pos = P + N * 0.02 + R * 0.04; // bias away from the source floor pixel
    for (int i=0;i<64;i++){
        pos += R * stepLen;
        stepLen *= 1.05;
        vec4 cs = uProj * vec4(pos,1.0);
        if (cs.w <= 0.0) break;
        vec2 uv = (cs.xy/cs.w)*0.5+0.5;
        if (uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0) break;
        float sceneZ = viewPos(uv).z;
        // Tight thickness window: ray must be just behind the stored surface.
        float dz = sceneZ - pos.z;
        if (dz > 0.001 && dz < 0.045){
            vec4 hitNR = texture(uNormalRough, uv);
            if (hitNR.a < 0.95) continue; // reject floor self-hits and curved contact objects
            vec3 hitN = normalize(hitNR.xyz * 2.0 - 1.0);
            if (dot(hitN, R) >= -0.03) continue;
            // fade near screen edges to hide SSR's hard cutoffs
            vec2 e = smoothstep(vec2(0.0), vec2(0.15), uv) *
                     smoothstep(vec2(0.0), vec2(0.15), 1.0-uv);
            float distFade = 1.0 - smoothstep(1.0, 4.0, length(pos - P));
            hitMask = e.x * e.y * distFade;
            return sampleReflection(uv);
        }
    }
    return vec3(0.0);
}

void main(){
    vec3 hdr = texture(uScene, vUV).rgb;
    vec4 nr  = texture(uNormalRough, vUV);

    // --- Screen-space reflections: floor only (rough < 0.3), subtle sheen ---
    if (uSSR == 1 && nr.a < 0.3) {
        vec3 N = normalize(nr.xyz * 2.0 - 1.0);
        vec3 P = viewPos(vUV);
        if (P.z < 0.0) {
            float mask;
            vec3 refl = ssr(P, N, mask);
            float NoV = clamp(dot(N, normalize(-P)), 0.0, 1.0);
            float fres = 0.10 + 0.35 * pow(1.0 - NoV, 4.0);
            float contact = contactFade(vUV, P.z);
            hdr = mix(hdr, refl, clamp(mask * fres * contact, 0.0, 0.30));
        }
    }

    // --- Bloom: bright-pass blur in a wider disk, softer threshold + stronger ---
    if (uPost == 1) {
        vec3 bloom = vec3(0.0); float wsum = 0.0;
        for (int i=0;i<16;i++){
            float a = float(i)/16.0 * 6.2831853;
            for (int r=1;r<=4;r++){
                vec2 off = vec2(cos(a),sin(a)) * (float(r)*4.5) / uRes;
                vec3 s = texture(uScene, vUV+off).rgb;
                vec3 bright = max(s - 2.2, 0.0);    // bloom only truly bright emitters
                float w = 1.0/float(r);
                bloom += bright * w; wsum += w;
            }
        }
        hdr += (bloom/max(wsum,1e-3)) * 0.45;
    }

    // --- Tonemap + gamma ---
    vec3 col = aces(hdr);
    col = pow(col, vec3(1.0/2.2));

    if (uPost == 1) {
        // Vignette.
        vec2 q = vUV - 0.5;
        col *= 1.0 - dot(q,q) * 0.55;
        // Subtle film grain.
        float g = hash12(vUV * uRes + uTime) - 0.5;
        col += g * 0.025;
    }

    // Dither to break 8-bit banding.
    col += (hash12(vUV * uRes) - 0.5) / 255.0;
    // Store luma in alpha so the FXAA pass can read edges cheaply.
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    FragColor = vec4(col, luma);
}
)";

// FXAA 3.11 (simplified) - anti-aliases the composited LDR image using the
// luma stored in alpha. Replaces the MSAA we lost when moving to the G-buffer.
static const char* kFxaaFrag = R"(#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;   // composited LDR color, luma in .a
uniform vec2 uRes;
uniform int uFXAA;

void main(){
    vec2 ip = 1.0 / uRes;
    if (uFXAA == 0) { FragColor = vec4(texture(uTex, vUV).rgb, 1.0); return; }

    float lM  = texture(uTex, vUV).a;
    float lNW = texture(uTex, vUV + vec2(-1,-1)*ip).a;
    float lNE = texture(uTex, vUV + vec2( 1,-1)*ip).a;
    float lSW = texture(uTex, vUV + vec2(-1, 1)*ip).a;
    float lSE = texture(uTex, vUV + vec2( 1, 1)*ip).a;

    float lMin = min(lM, min(min(lNW,lNE), min(lSW,lSE)));
    float lMax = max(lM, max(max(lNW,lNE), max(lSW,lSE)));
    float range = lMax - lMin;

    // Skip flat areas (no edge) -> keeps interiors crisp.
    if (range < max(0.0312, lMax * 0.125)) {
        FragColor = vec4(texture(uTex, vUV).rgb, 1.0);
        return;
    }

    vec2 dir;
    dir.x = -((lNW + lNE) - (lSW + lSE));
    dir.y =  ((lNW + lSW) - (lNE + lSE));
    float reduce = max((lNW+lNE+lSW+lSE) * 0.03125, 0.0078125);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp, -8.0, 8.0) * ip;

    vec3 rgbA = 0.5 * (texture(uTex, vUV + dir*(1.0/3.0-0.5)).rgb +
                       texture(uTex, vUV + dir*(2.0/3.0-0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(uTex, vUV + dir*-0.5).rgb +
                                     texture(uTex, vUV + dir* 0.5).rgb);
    float lB = dot(rgbB, vec3(0.299,0.587,0.114));
    FragColor = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
}
)";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compile error:\n%s\n", log);
        std::exit(1);
    }
    return s;
}

static GLuint makeProgram(const char* vsrc, const char* fsrc) {
    GLuint vs = compile(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Program link error:\n%s\n", log);
        std::exit(1);
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// ---------------------------------------------------------------------------
// Lightmap texture (HDR float).
// ---------------------------------------------------------------------------
static GLuint loadLightmapTexture(const std::string& path, GLuint existing = 0) {
    int w, h, n;
    float* data = stbi_loadf(path.c_str(), &w, &h, &n, 3);
    if (!data) {
        std::fprintf(stderr, "Failed to load lightmap '%s': %s\n",
                     path.c_str(), stbi_failure_reason());
        return existing;
    }
    GLuint tex = existing ? existing : 0;
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    stbi_image_free(data);
    std::printf("Loaded lightmap %s (%dx%d)\n", path.c_str(), w, h);
    return tex;
}

// Upload an HDR Vec3 pixel buffer (from an in-process bake) to a GL texture.
static GLuint uploadLightmapPixels(const std::vector<Vec3>& px, int res, GLuint existing = 0) {
    GLuint tex = existing ? existing : 0;
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    // Vec3 is three tightly-packed floats, so the buffer is GL_RGB / GL_FLOAT.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, res, res, 0, GL_RGB, GL_FLOAT, px.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

// ---------------------------------------------------------------------------
// NTC-decoded PBR material maps (8-bit, tiled across the sphere with mipmaps).
// 'srgb' selects an sRGB internal format for the color map so the shader works
// in linear space; data maps (normal/roughness/metalness/AO) stay linear.
// ---------------------------------------------------------------------------
static GLuint g_pbrAlbedo = 0, g_pbrNormal = 0, g_pbrRough = 0,
              g_pbrMetal = 0, g_pbrAO = 0;

static GLuint loadPBRTexture(const std::string& path, bool srgb) {
    int w, h, n;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 0);
    if (!data) {
        std::fprintf(stderr, "PBR: failed to load '%s': %s\n",
                     path.c_str(), stbi_failure_reason());
        return 0;
    }
    GLenum fmt = (n == 1) ? GL_RED : (n == 3) ? GL_RGB : GL_RGBA;
    GLenum ifmt = fmt;
    if (srgb) ifmt = (n == 3) ? GL_SRGB8 : GL_SRGB8_ALPHA8;
    GLuint tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // Single-channel maps replicate R across RGB so the shader can read .r.
    if (n == 1) {
        GLint sw[4] = { GL_RED, GL_RED, GL_RED, GL_ONE };
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, sw);
    }
    stbi_image_free(data);
    return tex;
}

// Load all five maps the shader uses from 'dir'. Returns false if albedo
// (the minimum) is missing. Maps the NTC decoder writes are named by semantic.
static bool loadPBRMaterial(const std::string& dir) {
    auto p = [&](const char* f) { return dir + "/" + f; };
    g_pbrAlbedo = loadPBRTexture(p("Color.png"), /*srgb=*/true);
    if (!g_pbrAlbedo) return false;
    g_pbrNormal = loadPBRTexture(p("Normal.png"), false);
    g_pbrRough  = loadPBRTexture(p("Roughness.png"), false);
    g_pbrMetal  = loadPBRTexture(p("Metalness.png"), false);
    g_pbrAO     = loadPBRTexture(p("AmbientOcclusion.png"), false);
    std::printf("Loaded NTC PBR material from %s\n", dir.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Fly camera.
// ---------------------------------------------------------------------------
struct Camera {
    // Default pose: standing INSIDE the box near the front-left corner,
    // looking diagonally toward the back-right. This sees the back wall, the
    // green (right) wall face-on, the floor and ceiling - and the red (left)
    // wall is right beside the camera. Fly with WASD to see the rest.
    glm::vec3 pos{ 0.30f, 0.62f, 1.35f };
    float yaw = -68.0f;    // toward -Z and +X (back-right)
    float pitch = -16.0f;  // angled down to show the glossy floor + reflections
    float speed = 1.2f;    // units/sec

    glm::vec3 front() const {
        float cy = std::cos(glm::radians(yaw)), sy = std::sin(glm::radians(yaw));
        float cp = std::cos(glm::radians(pitch)), sp = std::sin(glm::radians(pitch));
        return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
    }
};

static Camera g_cam;
static int    g_mode = 0;        // 0 = shaded, 1 = lightmap, 2 = UV debug
static bool   g_haveLightmap = false;
static bool   g_ssr = true;      // screen-space reflections
static bool   g_post = true;     // bloom/vignette/grain
static bool   g_texture = true;  // procedural surface texture/roughness
static bool   g_fxaa = true;     // FXAA anti-aliasing
static bool   g_spec = true;     // subtle specular highlight
static float  g_exposure = 1.4f;
static double g_lastX = 0, g_lastY = 0;
static bool   g_firstMouse = true;
static bool   g_captured = true;
static std::string g_lmPath;
static GLuint g_lmTex = 0;
static bool   g_reloadRequested = false;

// --- NTC-decoded PBR material on the sphere (the 'M' key) ---
// (texture handles g_pbrAlbedo..g_pbrAO are declared above the loaders.)
static std::string g_pbrDir;     // dir of decoded maps (empty = no PBR)
static bool   g_pbr = false;     // apply the material to the sphere
static bool   g_havePBR = false; // textures successfully loaded

// --- Offline screenshot mode (--shot FILE): frame the sphere, render a few
// settle frames, write a PNG, and exit. Used for deterministic verification.
static std::string g_shotPath;   // empty = interactive
static int    g_shotFrames = 8;  // render this many frames before capturing

// --- In-process bake state (the 'B' key) ---
enum class BakeState { Idle, Running, Done };
static std::atomic<BakeState> g_bakeState{ BakeState::Idle };
static std::atomic<bool>      g_bakeUploaded{ true };  // result consumed by GL thread?
static std::thread            g_bakeThread;
static Scene*                 g_scene = nullptr;        // owned by main
static Lightmap*              g_lm = nullptr;           // owned by main
static BakeSettings           g_bakeSettings = [] {
    BakeSettings s;
    s.spp = 64;        // low spp + OIDN denoise = fast, clean interactive bake
    s.bounces = 6;
    s.dilatePx = 8;    // more edge bleed for the higher-res atlas
    return s;
}();

// Kick off a bake on a background thread (GL-thread uploads the result later).
static void startBake() {
    if (g_bakeState.load() == BakeState::Running) return;
    if (g_bakeThread.joinable()) g_bakeThread.join();
    g_bakeState.store(BakeState::Running);
    g_bakeUploaded.store(false);
    g_bakeThread = std::thread([]() {
        bake(*g_scene, *g_lm, g_bakeSettings);   // fills g_lm->pixels()
        g_bakeState.store(BakeState::Done);
    });
}

static void mouseCb(GLFWwindow*, double x, double y) {
    if (!g_captured) { g_firstMouse = true; return; }
    if (g_firstMouse) { g_lastX = x; g_lastY = y; g_firstMouse = false; }
    float dx = (float)(x - g_lastX);
    float dy = (float)(g_lastY - y); // inverted: up is positive
    g_lastX = x; g_lastY = y;
    const float sens = 0.12f;
    g_cam.yaw   += dx * sens;
    g_cam.pitch += dy * sens;
    g_cam.pitch = std::min(89.0f, std::max(-89.0f, g_cam.pitch));
}

static void scrollCb(GLFWwindow*, double, double yoff) {
    g_cam.speed *= (yoff > 0 ? 1.15f : 0.87f);
    g_cam.speed = std::min(20.0f, std::max(0.05f, g_cam.speed));
}

static void keyCb(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) {
        if (g_captured) {
            g_captured = false;
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        } else {
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        }
    } else if (key == GLFW_KEY_F) {
        // Toggle lightmap on/off. Only switches to lightmap if we have one.
        if (g_mode == 1) {
            g_mode = 0;
        } else if (g_haveLightmap) {
            g_mode = 1;
        } else {
            std::printf("No lightmap yet - press B to bake one.\n");
        }
    } else if (key == GLFW_KEY_U) {
        g_mode = (g_mode == 2) ? 0 : 2;   // UV debug toggle
    } else if (key == GLFW_KEY_B) {
        if (g_bakeState.load() == BakeState::Running) {
            std::printf("Bake already in progress...\n");
        } else {
            std::printf("Baking (spp=%d, bounces=%d)...\n",
                        g_bakeSettings.spp, g_bakeSettings.bounces);
            startBake();
        }
    } else if (key == GLFW_KEY_R) {
        g_reloadRequested = true;
    } else if (key == GLFW_KEY_G) {
        g_ssr = !g_ssr;
        std::printf("Reflections (SSR): %s\n", g_ssr ? "ON" : "OFF");
    } else if (key == GLFW_KEY_P) {
        g_post = !g_post;
        std::printf("Post (bloom/vignette/grain): %s\n", g_post ? "ON" : "OFF");
    } else if (key == GLFW_KEY_T) {
        g_texture = !g_texture;
        std::printf("Surface texture/roughness: %s\n", g_texture ? "ON" : "OFF");
    } else if (key == GLFW_KEY_X) {
        g_fxaa = !g_fxaa;
        std::printf("FXAA: %s\n", g_fxaa ? "ON" : "OFF");
    } else if (key == GLFW_KEY_L) {
        g_spec = !g_spec;
        std::printf("Specular highlight: %s\n", g_spec ? "ON" : "OFF");
    } else if (key == GLFW_KEY_M) {
        if (g_havePBR) {
            g_pbr = !g_pbr;
            std::printf("NTC PBR material on sphere: %s\n", g_pbr ? "ON" : "OFF");
        } else {
            std::printf("No PBR material loaded (use --pbr <dir>; "
                        "decode it with scripts/decode_sphere_material.sh).\n");
        }
    }
}

static void mouseButtonCb(GLFWwindow* win, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !g_captured) {
        g_captured = true;
        g_firstMouse = true;
        glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

// ---------------------------------------------------------------------------
// Offscreen HDR G-buffer: color (RGBA16F) + view-normal/roughness + depth.
// ---------------------------------------------------------------------------
struct GBuffer {
    GLuint fbo = 0, color = 0, normalRough = 0, depth = 0;
    int w = 0, h = 0;

    void resize(int nw, int nh) {
        if (nw == w && nh == h && fbo) return;
        w = nw; h = nh;
        if (!fbo) glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        auto tex = [&](GLuint& t, GLint ifmt, GLenum fmt, GLenum type) {
            if (!t) glGenTextures(1, &t);
            glBindTexture(GL_TEXTURE_2D, t);
            glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        };
        tex(color,       GL_RGBA16F,            GL_RGBA,            GL_FLOAT);
        tex(normalRough, GL_RGBA16F,            GL_RGBA,            GL_FLOAT);
        tex(depth,       GL_DEPTH_COMPONENT24,  GL_DEPTH_COMPONENT, GL_FLOAT);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normalRough, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, depth, 0);
        GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, bufs);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

// Single-texture LDR target: holds the composited image (rgb + luma in .a) so
// the FXAA pass can sample neighbouring pixels.
struct ColorTarget {
    GLuint fbo = 0, tex = 0;
    int w = 0, h = 0;
    void resize(int nw, int nh) {
        if (nw == w && nh == h && fbo) return;
        w = nw; h = nh;
        if (!fbo) glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        if (!tex) glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

static void usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "  --lightmap FILE  HDR atlas to display on start (optional)\n"
        "  --res N          atlas resolution (default 1024)\n"
        "  --density F      texel density (default 220)\n"
        "  --spp N          samples per texel for in-viewer bake (default 64)\n"
        "  --bounces N      indirect bounces for in-viewer bake (default 6)\n"
        "  --exposure F     display exposure (default 1.4)\n"
        "  --pbr [DIR]      put NTC-decoded PBR material on the sphere\n"
        "                   (DIR defaults to assets/sphere_material)\n"
        "  --width N --height N   window size (default 1280x800)\n"
        "\nControls:\n"
        "  WASD / Space / Ctrl  move; mouse = look; scroll = speed\n"
        "  B = bake lightmap (in viewer)   F = lightmap on/off\n"
        "  G = reflections (SSR) on/off     P = bloom/vignette/grain on/off\n"
        "  T = surface texture on/off       X = FXAA anti-aliasing on/off\n"
        "  L = specular highlight on/off    U = UV-atlas debug view\n"
        "  M = NTC PBR material on sphere   R = reload --lightmap from disk\n"
        "  ESC = release mouse / quit\n",
        exe);
}

int main(int argc, char** argv) {
    std::string lmPath;            // empty = no preloaded lightmap; bake with B
    int res = 2048;
    int density = 360;
    int winW = 1280, winH = 800;

    auto needArg = [&](int& i) -> const char* {
        if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--lightmap") lmPath = needArg(i);
        else if (a == "--res") res = std::atoi(needArg(i));
        else if (a == "--density") density = std::atoi(needArg(i));
        else if (a == "--spp") g_bakeSettings.spp = std::atoi(needArg(i));
        else if (a == "--bounces") g_bakeSettings.bounces = std::atoi(needArg(i));
        else if (a == "--denoise") g_bakeSettings.denoise = true;
        else if (a == "--no-denoise") g_bakeSettings.denoise = false;
        else if (a == "--exposure") g_exposure = (float)std::atof(needArg(i));
        else if (a == "--width") winW = std::atoi(needArg(i));
        else if (a == "--height") winH = std::atoi(needArg(i));
        else if (a == "--pbr") {
            // Optional dir; default to the script's output location.
            if (i + 1 < argc && argv[i + 1][0] != '-') g_pbrDir = argv[++i];
            else g_pbrDir = "assets/sphere_material";
        }
        else if (a == "--shot") g_shotPath = needArg(i);
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    g_lmPath = lmPath;

    // Rebuild the exact scene + atlas layout used by the baker.
    Scene scene = buildCornellBox();
    Lightmap lm;
    if (!lm.build(scene, res, (float)density, 2)) {
        std::fprintf(stderr, "Atlas layout overflow at res=%d density=%d; "
                     "use the same values you baked with.\n", res, density);
        return 1;
    }
    std::vector<GLVertex> verts = buildVertices(scene, res);

    // Expose scene/lightmap to the bake thread (joined before main returns).
    g_scene = &scene;
    g_lm = &lm;

    // --- GLFW / GL setup ---
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // No window MSAA: we render to an offscreen HDR G-buffer for SSR/post.
    GLFWwindow* win = glfwCreateWindow(winW, winH, "Lightmap Viewer", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "window creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to init GLAD\n"); return 1;
    }

    glfwSetCursorPosCallback(win, mouseCb);
    glfwSetScrollCallback(win, scrollCb);
    glfwSetKeyCallback(win, keyCb);
    glfwSetMouseButtonCallback(win, mouseButtonCb);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glEnable(GL_DEPTH_TEST);
    // No backface culling: the camera flies both inside the box (seeing inner
    // wall faces) and outside it, so render every triangle double-sided.
    glDisable(GL_CULL_FACE);

    // --- buffers ---
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(GLVertex), verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(11 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(12 * sizeof(float)));
    glEnableVertexAttribArray(5);

    GLuint prog = makeProgram(kVert, kFrag);
    GLint locMVP = glGetUniformLocation(prog, "uMVP");
    GLint locExp = glGetUniformLocation(prog, "uExposure");
    GLint locMode = glGetUniformLocation(prog, "uMode");
    GLint locTex = glGetUniformLocation(prog, "uLightmap");
    GLint locAtlasRes = glGetUniformLocation(prog, "uAtlasRes");
    GLint locCam = glGetUniformLocation(prog, "uCamPos");
    GLint locN2V = glGetUniformLocation(prog, "uNormalToView");
    GLint locUseTex = glGetUniformLocation(prog, "uUseTexture");
    GLint locSpec = glGetUniformLocation(prog, "uSpec");
    GLint locLightPos = glGetUniformLocation(prog, "uLightPos");
    GLint locUsePBR = glGetUniformLocation(prog, "uUsePBR");
    GLint locPBRTiling = glGetUniformLocation(prog, "uPBRTiling");

    // Post/composite + FXAA programs (fullscreen passes) + a dummy VAO.
    GLuint post = makeProgram(kPostVert, kPostFrag);
    GLuint fxaa = makeProgram(kPostVert, kFxaaFrag);
    GLuint emptyVao; glGenVertexArrays(1, &emptyVao);
    GBuffer gbuf;
    ColorTarget ldr;

    // Always create a 1x1 placeholder texture so sampling is valid even pre-bake.
    glGenTextures(1, &g_lmTex);
    glBindTexture(GL_TEXTURE_2D, g_lmTex);
    { float black[3] = {0,0,0};
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 1, 1, 0, GL_RGB, GL_FLOAT, black);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); }

    // Optionally preload a baked atlas from disk.
    if (!g_lmPath.empty()) {
        GLuint t = loadLightmapTexture(g_lmPath, g_lmTex);
        if (t) { g_lmTex = t; g_haveLightmap = true; g_mode = 1; }
    }
    if (!g_haveLightmap)
        std::printf("No lightmap loaded - showing shaded view. Press B to bake.\n");

    // Optionally load the NTC-decoded PBR material for the sphere (--pbr).
    if (!g_pbrDir.empty()) {
        if (loadPBRMaterial(g_pbrDir)) {
            g_havePBR = true;
            g_pbr = true;     // on by default when supplied
            std::printf("Press M to toggle the NTC PBR material on the sphere.\n");
        } else {
            std::fprintf(stderr, "PBR: could not load material from '%s'. "
                         "Run scripts/decode_sphere_material.sh first.\n",
                         g_pbrDir.c_str());
        }
    }

    // Offline screenshot: aim the camera at the sphere (center 0.68,0.162,0.64,
    // radius 0.16) from a near, slightly-above angle so the material reads clearly.
    if (!g_shotPath.empty()) {
        g_captured = false;
        glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        glm::vec3 sphere(0.68f, 0.162f, 0.64f);
        g_cam.pos = sphere + glm::vec3(-0.22f, 0.14f, 0.32f);
        glm::vec3 dir = glm::normalize(sphere - g_cam.pos);
        g_cam.yaw = glm::degrees(std::atan2(dir.z, dir.x));
        g_cam.pitch = glm::degrees(std::asin(dir.y));
    }

    std::printf("Viewer ready. WASD + mouse to fly. B = bake, F = lightmap on/off, ESC = quit.\n");

    double last = glfwGetTime();
    int frameCount = 0;
    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dt = (float)(now - last);
        last = now;

        glfwPollEvents();

        // A background bake finished -> upload its pixels (GL must run here).
        if (g_bakeState.load() == BakeState::Done && !g_bakeUploaded.load()) {
            if (g_bakeThread.joinable()) g_bakeThread.join();
            g_lmTex = uploadLightmapPixels(g_lm->pixels(), g_lm->res(), g_lmTex);
            g_haveLightmap = true;
            g_mode = 1;                       // show the fresh bake immediately
            g_bakeUploaded.store(true);
            g_bakeState.store(BakeState::Idle);
            std::printf("Bake uploaded; showing lightmap.\n");
        }

        if (g_reloadRequested) {
            if (!g_lmPath.empty()) {
                GLuint t = loadLightmapTexture(g_lmPath, g_lmTex);
                if (t) { g_lmTex = t; g_haveLightmap = true; g_mode = 1; }
            } else {
                std::printf("No --lightmap path to reload (use B to bake).\n");
            }
            g_reloadRequested = false;
        }

        // Movement (only when captured).
        if (g_captured) {
            glm::vec3 f = g_cam.front();
            glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
            float v = g_cam.speed * dt;
            if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) g_cam.pos += f * v;
            if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) g_cam.pos -= f * v;
            if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) g_cam.pos -= r * v;
            if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) g_cam.pos += r * v;
            if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS) g_cam.pos.y += v;
            if (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) g_cam.pos.y -= v;
        }

        int fbw, fbh; glfwGetFramebufferSize(win, &fbw, &fbh);
        if (fbw < 1) fbw = 1; if (fbh < 1) fbh = 1;
        gbuf.resize(fbw, fbh);
        ldr.resize(fbw, fbh);

        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                                          (float)fbw / fbh, 0.02f, 100.0f);
        glm::mat4 view = glm::lookAt(g_cam.pos, g_cam.pos + g_cam.front(), glm::vec3(0, 1, 0));
        glm::mat4 mvp = proj * view;
        glm::mat3 normalToView = glm::mat3(view);  // view has no scale -> ok for normals

        // --- Pass 1: scene -> HDR G-buffer ---
        glBindFramebuffer(GL_FRAMEBUFFER, gbuf.fbo);
        glViewport(0, 0, fbw, fbh);
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix3fv(locN2V, 1, GL_FALSE, glm::value_ptr(normalToView));
        glUniform1f(locExp, g_exposure);
        glUniform1i(locMode, g_mode);
        glUniform1i(locUseTex, g_texture ? 1 : 0);
        glUniform1i(locSpec, g_spec ? 1 : 0);
        glUniform3f(locLightPos, 0.5f, 0.999f, 0.5f);   // ceiling light center
        glUniform3f(locCam, g_cam.pos.x, g_cam.pos.y, g_cam.pos.z);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_lmTex);
        glUniform1i(locTex, 0);
        glUniform1f(locAtlasRes, (float)res);

        // NTC PBR material samplers on units 1..5 (sphere only, when enabled).
        bool pbrOn = g_pbr && g_havePBR;
        glUniform1i(locUsePBR, pbrOn ? 1 : 0);
        glUniform1f(locPBRTiling, 3.0f);   // repeats across the sphere chart
        if (pbrOn) {
            struct { GLenum unit; GLuint tex; const char* name; } binds[] = {
                { GL_TEXTURE1, g_pbrAlbedo, "uPBRAlbedo" },
                { GL_TEXTURE2, g_pbrNormal, "uPBRNormal" },
                { GL_TEXTURE3, g_pbrRough,  "uPBRRough"  },
                { GL_TEXTURE4, g_pbrMetal,  "uPBRMetal"  },
                { GL_TEXTURE5, g_pbrAO,     "uPBRAO"     },
            };
            for (int i = 0; i < 5; ++i) {
                glActiveTexture(binds[i].unit);
                glBindTexture(GL_TEXTURE_2D, binds[i].tex);
                glUniform1i(glGetUniformLocation(prog, binds[i].name), 1 + i);
            }
        }
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());

        // --- Pass 2: composite (SSR + bloom + tonemap + grade) -> LDR target ---
        glBindFramebuffer(GL_FRAMEBUFFER, ldr.fbo);
        glViewport(0, 0, fbw, fbh);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(post);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, gbuf.color);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, gbuf.normalRough);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, gbuf.depth);
        glUniform1i(glGetUniformLocation(post, "uScene"), 0);
        glUniform1i(glGetUniformLocation(post, "uNormalRough"), 1);
        glUniform1i(glGetUniformLocation(post, "uDepth"), 2);
        glUniform2f(glGetUniformLocation(post, "uRes"), (float)fbw, (float)fbh);
        glUniform1f(glGetUniformLocation(post, "uExposure"), g_exposure);
        glUniformMatrix4fv(glGetUniformLocation(post, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
        glUniformMatrix4fv(glGetUniformLocation(post, "uInvProj"), 1, GL_FALSE,
                           glm::value_ptr(glm::inverse(proj)));
        glUniform1i(glGetUniformLocation(post, "uSSR"), g_ssr ? 1 : 0);
        glUniform1i(glGetUniformLocation(post, "uPost"), g_post ? 1 : 0);
        glUniform1f(glGetUniformLocation(post, "uTime"), (float)now);
        glBindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // --- Pass 3: FXAA -> screen ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbw, fbh);
        glUseProgram(fxaa);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, ldr.tex);
        glUniform1i(glGetUniformLocation(fxaa, "uTex"), 0);
        glUniform2f(glGetUniformLocation(fxaa, "uRes"), (float)fbw, (float)fbh);
        glUniform1i(glGetUniformLocation(fxaa, "uFXAA"), g_fxaa ? 1 : 0);
        glBindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(win);

        // Offline screenshot: after a few settle frames, read the back buffer
        // (the FXAA pass blits to the default framebuffer) and write a PNG.
        if (!g_shotPath.empty() && ++frameCount >= g_shotFrames) {
            std::vector<unsigned char> px((size_t)fbw * fbh * 3);
            glReadBuffer(GL_BACK);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, fbw, fbh, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            stbi_flip_vertically_on_write(1);   // GL origin is bottom-left
            if (stbi_write_png(g_shotPath.c_str(), fbw, fbh, 3, px.data(), fbw * 3))
                std::printf("Wrote screenshot %s (%dx%d)\n", g_shotPath.c_str(), fbw, fbh);
            else
                std::fprintf(stderr, "Failed to write screenshot %s\n", g_shotPath.c_str());
            break;
        }
    }

    if (g_bakeThread.joinable()) g_bakeThread.join();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
