#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
};

class Ray {
public:
    __host__ __device__ Ray::Ray()
        : origin(), direction()
    {}
    __host__ __device__ Ray::Ray(const Point3f& o, const Vector3f& d)
        : origin(o), direction(d)
    {}

    __host__ __device__ Point3f getOrigin() const
    {
        return origin;
    }

    __host__ __device__ Vector3f getDirection() const
    {
        return direction;
    }

    //  Falls slightly short so that it doesn't intersect the object it's hitting. 
    __host__ __device__ Point3f evaluate(float t) {
        return origin + (t - .0001f) * glm::normalize(direction);
    }
public:
    Point3f origin;
    Vector3f direction;
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
    // EYE
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
    // DEPTH in Camera
    int traceDepth;
    std::vector<glm::vec3> image;
    // FILE in Camera
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
  glm::vec3 intersectionPoint;
};

struct isTerminated
{
    __host__ __device__
    bool operator()(const PathSegment& p)
    {
        return p.remainingBounces > 0;
    }
};

struct sortMaterial 
{
    __host__ __device__ 
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};