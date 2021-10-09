#pragma once

#include <string>
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Transform
{
    glm::vec3 translate;
    glm::vec3 rotate;
    glm::vec3 scale;
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
    // for triangle
    glm::vec3 pos1;
    glm::vec3 pos2;
    glm::vec3 pos3; 
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

struct Triangle
{
    glm::vec3 p1, p2, p3;
    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) : p1(p1), p2(p2), p3(p3) {}
    glm::vec3& operator[](int i)
    {
        if (i > 2)
            return p1;
        else
        {
            if (i == 0)
                return p1;
            else if (i == 1)
                return p2;
            else
                return p3;
        }
    }
};

struct BVHNode
{
    BVHNode() : leftChild(-1), rightChild(-1), axis(0) {}
    ~BVHNode() {}

    int leftChild; 
    int rightChild; 
    unsigned int axis; // Which axis split this node represents
    Geom boundingBox;
    Geom triangle; // only initialized if this is a leaf node

};

void buildTree(
    std::vector<BVHNode>* bvhNodes,
    std::vector<Triangle>* primitives,
    int node, int primStart, int primEnd);

struct BVHTree
{
    BVHTree(int inx, Transform& transform, int materialId) : bvhNodes(inx), transform(transform), materialId(materialId)
    {}
    ~BVHTree() {}

    void build(std::vector<BVHNode>* bvhNodes, std::vector<Triangle>* primitives, int startInx, int endInx)
    {
        BVHNode root = BVHNode();
        root.axis = 0;
        bvhNodes->push_back(root);
        buildTree(bvhNodes, primitives, bvhNodes->size() - 1, startInx, endInx);
    }

    Transform transform;
    int bvhNodes;
    int materialId;
};
