#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "tiny_gltf.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Mesh {
    int count;
    int i_offset;
    int v_offset;
    int n_offset;
    int uv_offset;
    glm::vec3 bbox_max;
    glm::vec3 bbox_min;
    glm::mat4 pivot_xform;
};

struct MeshData {
    Mesh* meshes;
    glm::vec3* vertices;
    glm::vec3* normals;
    uint16_t* indices;
    glm::vec2* uvs;

    void free() {
      cudaFree(meshes);
      cudaFree(vertices);
      cudaFree(normals);
      cudaFree(indices);
      cudaFree(uvs);
    }
};

struct Geom {
    enum GeomType type;
    int materialid;
    int meshid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Texture {
  int test;
};

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
    int spp;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
};

// Predicate for checking if a path is complete or not
struct isPathCompleted {
  __host__ __device__
    bool operator()(const PathSegment& pathSegment) {
    return pathSegment.remainingBounces <= 0;
  }
};

struct compareIntersections {
  __host__ __device__
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
    return a.materialId < b.materialId;
  }
};