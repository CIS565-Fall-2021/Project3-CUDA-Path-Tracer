#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#define BACKGROUND_COLOR (glm::vec3(0.0f))

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 points[3];
    glm::vec3 normal;
    glm::vec3 color;
};

struct Node {
    Node(glm::vec3 pos, glm::vec3 dim, int i, bool leaf) :
        position(pos), dimension(dim), childrenStartIndex(i), leaf(leaf),
        host_triangles(new std::vector<Triangle>),
        device_triangles(NULL), numTriangles(0) {};
    glm::vec3 position;
    glm::vec3 dimension;
    bool contains(Triangle& triangle);
    void computeChildLocations(glm::vec3* childLocations);
    bool triangleIntersectionTest(Triangle& triangle);
   
    // If false, then childrenStartIndex points to first of 8 Nodes in tree array.
    // If true, then numTriangles, host_triangles, and device_triangles are populated.
    bool leaf;

    int childrenStartIndex;    
    std::vector<Triangle>* host_triangles;
    Triangle* device_triangles;
    int numTriangles;

    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum GeomType {
    SPHERE,
    CUBE,
    BB
};

struct Geom {
    Geom() : host_tree(new std::vector<Node>) {}
    enum GeomType type;
    int materialid;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    glm::vec3 minDims;
    glm::vec3 maxDims;

    char filename;

    std::vector<Triangle>* host_triangles;
    Triangle* device_triangles;
    int numTriangles;

    std::vector<Node>* host_tree;
    Node* device_tree;
    int treeSize;
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
  bool outside;
  glm::vec3 surfaceNormal;
  glm::vec3 intersectionPt;
  int materialId;
  __host__ __device__ bool operator < (const ShadeableIntersection& si) const {
      return materialId < si.materialId;
  }
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
};