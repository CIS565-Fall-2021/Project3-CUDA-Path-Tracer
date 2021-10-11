#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle
{
    glm::vec3 pos[3];
    glm::vec3 normal[3];
    glm::vec2 uv[3];
    glm::vec4 tangent[3];
};

struct AABB 
{
    glm::vec3 bound[2] = { glm::vec3(FLT_MAX), glm::vec3(FLT_MIN) };
};

struct Geom 
{
    enum GeomType type;
    int materialid;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int triBeginIdx;
    int triEndIdx;
    AABB aabb;
};

struct TexInfo
{
    int offset = -1;
    int width;
    int height;
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
    TexInfo tex;
    TexInfo bump;
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
    float focalDist;
    float aperture;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

struct pathRemains
{
    __device__ bool operator()(const PathSegment& pathSeg)
    {
        return pathSeg.remainingBounces > 0;
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection 
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;

  __host__ __device__ bool operator<(const ShadeableIntersection& other) const
  {
      return materialId < other.materialId;
  }
};
