#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "scenestruct/material.h"
#include "scenestruct/geometry.h"

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

    int recordDepth = -1;
};

struct GBufferData {
    i32 geometryId = -1;
    i32 materialId = -1;
    i32 stencilId = -1;
    glm::vec3 baseColor;
    glm::vec3 normal;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;

    GBufferData gBufferData;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    __host__ __device__ bool operator<(const ShadeableIntersection& s) const {
        return materialId < s.materialId;
    }

    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
    //glm::vec3 barycentric;
    int geometryId = -1;
    int materialId = -1;
    int stencilId = -1;
};
