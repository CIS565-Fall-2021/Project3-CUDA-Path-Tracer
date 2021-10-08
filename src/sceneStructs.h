#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

// unused, maybe delete
#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define STBI_NO_FAILURE_STRINGS
#define STBI_FAILURE_USERMSG

// Flags for different optimizations and timing

//! At most one debug flag at a time
// try loading texture base color onto cube
// #define DEBUG_TEX_BASE_COLOR
// Sets color to the surface normal for debug
#define DEBUG_SURFACE_NORMAL
// Sets to grayscale representing the t value
// #define DEBUG_T_VAL
// Times execution of whole pathtrace, assumes memops time << computation time
// #define TIME_PATHTRACE
// Groups the rays by material type for better warp coherence (stream compact)
#define GROUP_RAYS
// Removes finished rays
#define COMPACT_RAYS
// Jitter the ray directions slightly to trade noise for jagged edges
#define ANTIALIASING
// Use thin lens to randomize ray origin to approximate depth of field
// #define DEPTH_OF_FIELD
#if defined(ANTIALIASING) || defined(DEPTH_OF_FIELD)
#else
// Cache first iter; only if first rays cast are deterministic
#define CACHE_FIRST
#endif
// Use AABBs to quick check mesh intersections
#define BV_CULL

#define SMALL_OFFSET 0.001f
#define OFFSET_VECTOR(newDir) SMALL_OFFSET *newDir

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
    MESH
};

struct Ray
{
    glm::vec3 origin = glm::vec3();
    glm::vec3 direction = glm::vec3();
};

struct Triangle
{
    glm::vec3 pos[3] = {glm::vec3(), glm::vec3(), glm::vec3()};
    glm::vec3 norm[3] = {glm::vec3(), glm::vec3(), glm::vec3()};
    glm::vec2 uv[3] = {glm::vec2(), glm::vec2(), glm::vec2()};
    // To go from tan space to obj space, col(t), col(b), col(n)
    glm::vec3 tangent = glm::vec3();
    glm::vec3 bitangent = glm::vec3();
    glm::vec3 planarNorm = glm::vec3();
};

struct TexData
{
    uint8_t bCol[3];
    uint8_t bump[3];
    uint8_t amOc;
    uint8_t rogh;
    uint8_t metl;
    bool emit;
};

// Mesh knows its triangles startindex in triangle buffer and how many triangles it has
struct Geom
{
    enum GeomType type = MESH;
    int materialid = -1;
    glm::vec3 translation = glm::vec3();
    glm::vec3 rotation = glm::vec3();
    glm::vec3 scale = glm::vec3();
    glm::mat4 transform = glm::mat4();
    glm::mat4 inverseTransform = glm::mat4();
    glm::mat4 invTranspose = glm::mat4();
    // std::vector<struct Triangle> ts;
    glm::vec3 min = glm::vec3(0.f);
    glm::vec3 max = glm::vec3(0.f);
    bool useTexture = false;
    // int meshIdx = -1;
    int triIdx = -1;
    int numTris = 0;
};

struct Material
{
    glm::vec3 color = glm::vec3(0.7241664348743744, 0.38082699022272768, 0.00178208301015399);
    struct
    {
        float exponent = 0.f;
        glm::vec3 color = glm::vec3();
    } specular;
    int texHeight;
    int texWidth;
    float hasReflective = 0.f;
    float hasRefractive = 0.f;
    float indexOfRefraction = 0.f;
    float emittance = 0.f;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position = glm::vec3();
    glm::vec3 lookAt = glm::vec3();
    glm::vec3 view = glm::vec3();
    glm::vec3 up = glm::vec3();
    glm::vec3 right = glm::vec3();
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations = 0;
    int traceDepth = -1;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color = glm::vec3();
    int pixelIndex = -1;
    int remainingBounces = -1;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t = 0.f;
    glm::vec3 surfaceNormal = glm::vec3();
    int materialId = -1;
    glm::vec2 uvs = glm::vec2(0.f);
    bool useTexture = false;
};
