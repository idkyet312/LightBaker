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

// ---------------------------------------------------------------------------
// Interleaved vertex: world position + atlas UV + normal + albedo.
// ---------------------------------------------------------------------------
struct GLVertex { float px, py, pz; float u, v; float nx, ny, nz; float ar, ag, ab; };

// Map a triangle's chart-local UV (0..1 within its quad) to atlas UV (0..1
// across the whole atlas), matching how the baker/preview place charts.
static void chartUVtoAtlas(const Quad& q, int res, float lu, float lv,
                           float& au, float& av) {
    float texX = q.chartX + lu * q.chartW;
    float texY = q.chartY + lv * q.chartH;
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
        for (int i = 0; i < 3; ++i) {
            float au, av;
            chartUVtoAtlas(q, res, uv[i].x, uv[i].y, au, av);
            verts.push_back({ p[i].x, p[i].y, p[i].z, au, av,
                              t.ng.x, t.ng.y, t.ng.z,
                              alb.x, alb.y, alb.z });
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
uniform mat4 uMVP;
out vec2 vUV;
out vec3 vNormal;
out vec3 vWorldPos;
out vec3 vAlbedo;
void main() {
    vUV = aUV;
    vNormal = aNormal;
    vWorldPos = aPos;
    vAlbedo = aAlbedo;
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

layout(location=0) out vec4 oColor;      // linear HDR radiance
layout(location=1) out vec4 oNormalRough; // viewNormal.xyz (0..1), roughness in .a

uniform sampler2D uLightmap;
uniform float uExposure;
uniform int uMode;          // 0 = shaded, 1 = lightmap, 2 = UV debug
uniform vec3 uCamPos;
uniform mat3 uNormalToView; // world->view for normals
uniform int uUseTexture;    // procedural surface texture/roughness on/off

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

    // Procedural surface variation: faint multi-octave noise modulating albedo,
    // and a roughness field so different surfaces reflect differently.
    float rough = 0.6;
    if (uUseTexture == 1) {
        float n = vnoise(vWorldPos * 18.0) * 0.6 + vnoise(vWorldPos * 60.0) * 0.4;
        albedo *= (0.93 + 0.07 * n);                 // subtle tonal variation
        rough = clamp(0.35 + 0.25 * n, 0.05, 0.9);   // varied micro-roughness
    }
    // The floor (normal ~ +Y) is glossier so reflections read clearly.
    if (N.y > 0.7) rough = min(rough, 0.18);

    vec3 col;
    if (uMode == 1) {
        vec3 irr = texture(uLightmap, vUV).rgb;
        col = irr * albedo * uExposure;              // linear HDR (no tonemap here)
    } else if (uMode == 2) {
        col = vec3(vUV, 0.0);
        rough = 1.0;
    } else {
        vec3 L = normalize(uCamPos - vWorldPos);
        float diff = abs(dot(N, L));
        col = albedo * (0.25 + 0.75 * diff) * uExposure;
        rough = 1.0;
    }

    oColor = vec4(col, 1.0);
    vec3 vN = normalize(uNormalToView * N);
    oNormalRough = vec4(vN * 0.5 + 0.5, rough);
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

// Screen-space reflection ray-march; returns reflected color and a hit mask.
vec3 ssr(vec3 P, vec3 N, out float hitMask){
    hitMask = 0.0;
    vec3 V = normalize(P);             // view dir (camera at origin in view space)
    vec3 R = reflect(V, N);
    if (R.z > 0.0) return vec3(0.0);   // reflecting toward camera plane: skip
    float stepLen = 0.025;
    vec3 pos = P;
    for (int i=0;i<48;i++){
        pos += R * stepLen;
        stepLen *= 1.07;               // grow step for range
        vec4 cs = uProj * vec4(pos,1.0);
        if (cs.w <= 0.0) break;
        vec2 uv = (cs.xy/cs.w)*0.5+0.5;
        if (uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0) break;
        float sceneZ = viewPos(uv).z;
        if (pos.z < sceneZ - 0.002 && pos.z > sceneZ - 0.25){
            hitMask = 1.0;
            // fade near screen edges to hide SSR's edge artifacts
            vec2 e = smoothstep(vec2(0.0), vec2(0.12), uv) *
                     smoothstep(vec2(0.0), vec2(0.12), 1.0-uv);
            hitMask *= e.x*e.y;
            return texture(uScene, uv).rgb;
        }
    }
    return vec3(0.0);
}

void main(){
    vec3 hdr = texture(uScene, vUV).rgb;
    vec4 nr  = texture(uNormalRough, vUV);

    // --- Screen-space reflections (Fresnel-weighted, attenuated by roughness) ---
    if (uSSR == 1 && nr.a < 0.5) {
        vec3 N = normalize(nr.xyz * 2.0 - 1.0);
        vec3 P = viewPos(vUV);
        if (P.z < 0.0) {
            float mask;
            vec3 refl = ssr(P, N, mask);
            float NoV = clamp(dot(N, normalize(-P)), 0.0, 1.0);
            float fres = 0.04 + 0.96 * pow(1.0 - NoV, 5.0);     // Schlick
            float gloss = 1.0 - nr.a / 0.5;                     // glossier -> stronger
            hdr = mix(hdr, refl, mask * fres * clamp(gloss,0.0,1.0) * 0.9);
        }
    }

    // --- Bloom: cheap bright-pass blur sampled in a small disk ---
    if (uPost == 1) {
        vec3 bloom = vec3(0.0); float wsum = 0.0;
        for (int i=0;i<12;i++){
            float a = float(i)/12.0 * 6.2831853;
            for (int r=1;r<=2;r++){
                vec2 off = vec2(cos(a),sin(a)) * (float(r)*3.5) / uRes;
                vec3 s = texture(uScene, vUV+off).rgb;
                vec3 bright = max(s - 1.0, 0.0);    // only HDR > 1 blooms
                float w = 1.0/float(r);
                bloom += bright * w; wsum += w;
            }
        }
        hdr += (bloom/max(wsum,1e-3)) * 0.5;
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
    FragColor = vec4(col, 1.0);
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
static float  g_exposure = 1.4f;
static double g_lastX = 0, g_lastY = 0;
static bool   g_firstMouse = true;
static bool   g_captured = true;
static std::string g_lmPath;
static GLuint g_lmTex = 0;
static bool   g_reloadRequested = false;

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

static void usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "  --lightmap FILE  HDR atlas to display on start (optional)\n"
        "  --res N          atlas resolution (default 1024)\n"
        "  --density F      texel density (default 220)\n"
        "  --spp N          samples per texel for in-viewer bake (default 64)\n"
        "  --bounces N      indirect bounces for in-viewer bake (default 6)\n"
        "  --exposure F     display exposure (default 1.4)\n"
        "  --width N --height N   window size (default 1280x800)\n"
        "\nControls:\n"
        "  WASD / Space / Ctrl  move; mouse = look; scroll = speed\n"
        "  B = bake lightmap (in viewer)   F = lightmap on/off\n"
        "  G = reflections (SSR) on/off     P = bloom/vignette/grain on/off\n"
        "  T = surface texture on/off       U = UV-atlas debug view\n"
        "  R = reload --lightmap from disk  ESC = release mouse / quit\n",
        exe);
}

int main(int argc, char** argv) {
    std::string lmPath;            // empty = no preloaded lightmap; bake with B
    int res = 1024;
    int density = 220;
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

    GLuint prog = makeProgram(kVert, kFrag);
    GLint locMVP = glGetUniformLocation(prog, "uMVP");
    GLint locExp = glGetUniformLocation(prog, "uExposure");
    GLint locMode = glGetUniformLocation(prog, "uMode");
    GLint locTex = glGetUniformLocation(prog, "uLightmap");
    GLint locCam = glGetUniformLocation(prog, "uCamPos");
    GLint locN2V = glGetUniformLocation(prog, "uNormalToView");
    GLint locUseTex = glGetUniformLocation(prog, "uUseTexture");

    // Post/composite program (fullscreen pass) + a dummy VAO to drive it.
    GLuint post = makeProgram(kPostVert, kPostFrag);
    GLuint emptyVao; glGenVertexArrays(1, &emptyVao);
    GBuffer gbuf;

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

    std::printf("Viewer ready. WASD + mouse to fly. B = bake, F = lightmap on/off, ESC = quit.\n");

    double last = glfwGetTime();
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
        glUniform3f(locCam, g_cam.pos.x, g_cam.pos.y, g_cam.pos.z);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_lmTex);
        glUniform1i(locTex, 0);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());

        // --- Pass 2: composite (SSR + bloom + tonemap + grade) -> screen ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

        glfwSwapBuffers(win);
    }

    if (g_bakeThread.joinable()) g_bakeThread.join();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
