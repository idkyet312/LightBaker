#include "denoiser.h"
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef USE_OIDN
#include <OpenImageDenoise/oidn.hpp>

bool denoiseLightmap(Lightmap& lm, const Scene& scene) {
    const int res = lm.res();
    const size_t n = (size_t)res * res;

    // Aux buffers guide edge-preserving filtering. We denoise *irradiance*
    // (the stored lighting), keeping surface albedo crisp since it is applied
    // separately at display time.
    std::vector<Vec3> albedo = lm.albedoAtlas(scene);
    std::vector<Vec3> normal = lm.normalAtlas(scene);
    std::vector<Vec3>& color = lm.pixelsMut();      // in/out: irradiance

    oidn::DeviceRef device = oidn::newDevice();      // picks CPU device by default
    device.commit();
    const char* err;
    if (device.getError(err) != oidn::Error::None) {
        std::printf("OIDN device error: %s\n", err ? err : "?");
        return false;
    }

    // OIDN 2.x requires device-accessible storage: allocate device buffers and
    // copy our std::vectors in/out (raw host pointers are rejected on some builds).
    const size_t bytes = n * sizeof(Vec3);
    auto makeBuf = [&](const std::vector<Vec3>& src) {
        oidn::BufferRef b = device.newBuffer(bytes);
        std::memcpy(b.getData(), src.data(), bytes);
        return b;
    };
    oidn::BufferRef colorBuf  = makeBuf(color);
    oidn::BufferRef albedoBuf = makeBuf(albedo);
    oidn::BufferRef normalBuf = makeBuf(normal);
    oidn::BufferRef outBuf    = device.newBuffer(bytes);

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color",  colorBuf,  oidn::Format::Float3, res, res);
    filter.setImage("albedo", albedoBuf, oidn::Format::Float3, res, res);
    filter.setImage("normal", normalBuf, oidn::Format::Float3, res, res);
    filter.setImage("output", outBuf,    oidn::Format::Float3, res, res);
    filter.set("hdr", true);            // irradiance is HDR / unbounded
    filter.set("cleanAux", true);       // albedo & normal are noise-free
    filter.commit();
    filter.execute();

    if (device.getError(err) != oidn::Error::None) {
        std::printf("OIDN filter error: %s\n", err ? err : "?");
        return false;
    }

    std::memcpy(color.data(), outBuf.getData(), bytes);  // denoised -> lightmap
    std::printf("Denoised lightmap (OIDN, %dx%d)\n", res, res);
    return true;
}

#else  // USE_OIDN not defined

bool denoiseLightmap(Lightmap&, const Scene&) {
    std::printf("Denoise requested but OIDN was not compiled in.\n");
    return false;
}

#endif
