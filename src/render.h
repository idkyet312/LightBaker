// render.h - pinhole-camera preview that samples the *baked* lightmap.
#pragma once
#include "scene.h"
#include "lightmap.h"
#include <vector>

// Render an (w x h) preview image by casting primary rays and looking up the
// baked atlas at each hit's UV2. Returns an HDR RGB buffer (row-major).
std::vector<Vec3> renderPreview(const Scene& scene, const Lightmap& lm,
                                int w, int h, int aa);
