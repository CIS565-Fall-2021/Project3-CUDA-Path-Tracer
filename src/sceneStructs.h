#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "tiny_gltf.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

typedef glm::vec3 Color;

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
    int n_offset = -1;
    int uv_offset = -1;
    int t_offset = -1;
    int mat_id;
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
    glm::vec4* tangents;

    void free() {
      cudaFree(meshes);
      cudaFree(vertices);
      cudaFree(normals);
      cudaFree(indices);
      cudaFree(uvs);
      cudaFree(tangents);
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

struct Texture {
  int width;
  int height;
  int components;
  unsigned char* image;
  int size;
};

struct TextureInfo {
  int index = -1;
  int texCoord;

  TextureInfo& operator=(const tinygltf::TextureInfo t) {
    index = t.index;
    texCoord = t.texCoord;
    return *this;
  }
};

struct NormalTextureInfo : TextureInfo {
  float scale = 1.0f;

  NormalTextureInfo& operator=(const tinygltf::NormalTextureInfo t) {
    index = t.index;
    texCoord = t.texCoord;
    scale = t.scale;
    return *this;
  }
};

// Based on tinygltf::PbrMetallicRoughness
// pbrMetallicRoughness class defined in glTF 2.0 spec.
// Defining a custom PbrMetallicRoughness struct here because
// all vectors have to use glm::vec to be supported in CUDA
struct PbrMetallicRoughness {
  Color baseColorFactor = Color(1.0f);  // Change to vec4 if alpha is used
  TextureInfo baseColorTexture;
  float metallicFactor;   // default 1
  float roughnessFactor;  // default 1
  TextureInfo metallicRoughnessTexture;

  __host__ __device__ PbrMetallicRoughness()
    : baseColorFactor(Color(1.0f)),
      metallicFactor(1.0f),
      roughnessFactor(1.0f) {}
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

    PbrMetallicRoughness pbrMetallicRoughness;

    NormalTextureInfo normalTexture;
    //tinygltf::OcclusionTextureInfo occlusionTexture;
    //tinygltf::TextureInfo emissiveTexture;
    glm::vec3 emissiveFactor;  // length 3. default [0, 0, 0]
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
    Color color;
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
  glm::vec4 tangent;
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
    return a.materialId > b.materialId;
  }
};