cmake_minimum_required(VERSION 3.15)
project(LightBaker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Threads REQUIRED)

# --- Optional Intel Open Image Denoise (prebuilt SDK in third_party/oidn) ---
set(OIDN_DIR "${CMAKE_SOURCE_DIR}/third_party/oidn")
set(OIDN_ENABLED FALSE)
if(EXISTS "${OIDN_DIR}/include/OpenImageDenoise/oidn.hpp")
    set(OIDN_ENABLED TRUE)
    message(STATUS "denoiser: ENABLED (OIDN SDK at ${OIDN_DIR})")
else()
    message(STATUS "denoiser: DISABLED (no OIDN SDK in third_party/oidn)")
endif()

# --- Shared scene / baking code used by both the baker and the viewer ---
add_library(bakerlib STATIC
    src/bvh.cpp
    src/scene.cpp
    src/lightmap.cpp
    src/pathtracer.cpp
    src/render.cpp
    src/denoiser.cpp
)
target_include_directories(bakerlib PUBLIC src third_party)
target_link_libraries(bakerlib PUBLIC Threads::Threads)
if(OIDN_ENABLED)
    target_compile_definitions(bakerlib PUBLIC USE_OIDN)
    target_include_directories(bakerlib PUBLIC "${OIDN_DIR}/include")
    target_link_libraries(bakerlib PUBLIC "${OIDN_DIR}/lib/OpenImageDenoise.lib")
endif()
if(MSVC)
    target_compile_options(bakerlib PRIVATE /O2 /fp:fast)
else()
    target_compile_options(bakerlib PRIVATE -O3 -ffast-math)
endif()

# Copy the OIDN runtime DLLs next to an executable target.
function(copy_oidn_dlls tgt)
    if(OIDN_ENABLED AND WIN32)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${OIDN_DIR}/bin" "$<TARGET_FILE_DIR:${tgt}>"
            COMMENT "Copying OIDN runtime DLLs next to ${tgt}")
    endif()
endfunction()

# --- Offline baker (CLI) ---
add_executable(baker
    src/main.cpp
    src/baker.cpp
)
target_link_libraries(baker PRIVATE bakerlib)
if(MSVC)
    target_compile_options(baker PRIVATE /O2 /fp:fast)
else()
    target_compile_options(baker PRIVATE -O3 -ffast-math)
endif()
copy_oidn_dlls(baker)

# --- Real-time viewer (OpenGL). Built only if GLFW/GLAD/GLM are available. ---
find_package(glfw3 CONFIG QUIET)
find_package(glad CONFIG QUIET)
find_package(glm CONFIG QUIET)

if(glfw3_FOUND AND glad_FOUND AND glm_FOUND)
    add_executable(viewer src/viewer.cpp src/baker.cpp)
    target_link_libraries(viewer PRIVATE bakerlib glfw glad::glad glm::glm)
    if(WIN32)
        target_link_libraries(viewer PRIVATE opengl32)
    endif()
    copy_oidn_dlls(viewer)
    message(STATUS "viewer: ENABLED (glfw3 + glad + glm found)")
else()
    message(STATUS "viewer: DISABLED (need glfw3, glad, glm; e.g. via vcpkg). "
                   "Configure with -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake")
endif()
