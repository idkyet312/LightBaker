// denoiser.h - optional Intel Open Image Denoise (OIDN) wrapper.
#pragma once
#include "scene.h"
#include "lightmap.h"

// Denoise the lightmap's HDR irradiance in place, using albedo + normal atlases
// as edge-preserving auxiliary buffers. No-op (returns false) if OIDN support
// was not compiled in. Only covered texels are touched; run before dilation.
bool denoiseLightmap(Lightmap& lm, const Scene& scene);
