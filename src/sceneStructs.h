#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType { SPHERE, CUBE, TRIANGLE };

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Geom {
  enum GeomType type;
  int materialid;
  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inverseTransform;
  glm::mat4 invTranspose;
  struct Triangle {
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
  } triangle;
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

struct Camera {
  glm::ivec2 resolution;
  glm::vec3 position;
  glm::vec3 lookAt;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec2 fov;
  glm::vec2 pixelLength;
  float focalDistance;  // distance from film to plane of focus
  float lensRadius;
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

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

// CHECKITOUT - a simple struct for storing scene geometry information
// per-pixel. What information might be helpful for guiding a denoising filter?
struct GBufferPixel {
  float t;
};
