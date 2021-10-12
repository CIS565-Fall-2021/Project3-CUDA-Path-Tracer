#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define DOF_APERATURE       1.0f
#define DOF_FOCUSDIST       2.0f
#define USE_DOF             0
#define USE_BOUNDING_BOX    1
#define USE_MESH_LOADING    1
#define ANTIALIASING        0
#define SCHLICK             0
#define USE_CACHE           1
#define USE_SORT            1
#define USE_PARTITION       1
#define USE_REMOVE_IF       0
#define PRINT               0
#define ERRORCHECK          0

enum GeomType {
    SPHERE,
    CUBE,
    OBJ,
};

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

    int  triCount = 0; 
    glm::vec4* host_VecNorArr; 
    glm::vec4* dev_VecNorArr;

    glm::mat4 bbInverseTransform;
    glm::vec3 bbScale;
    
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

    float aperture;
    float focusDist; 

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
