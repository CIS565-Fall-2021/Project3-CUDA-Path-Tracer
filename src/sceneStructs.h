#pragma once

#include <string>
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include "glm/glm.hpp"

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


struct KDNode
{
    KDNode() : leftChild(nullptr), rightChild(nullptr), axis(0), minCorner(), maxCorner(), particles() {}
    ~KDNode() {
        delete leftChild;
        delete rightChild;
    }

    KDNode* leftChild;
    KDNode* rightChild;
    unsigned int axis; // Which axis split this node represents
    glm::vec3 minCorner, maxCorner; // The world-space bounds of this node
    std::vector<std::array<glm::vec3, 3>> particles; // A collection of pointers to the particles contained in this node.
};

// TODO: use getMinVertex to clean up code
bool xSort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b);
bool ySort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b);
bool zSort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b);

void buildTree(
    KDNode* node, 
    std::vector<std::array<glm::vec3, 3>>& triangles, 
    std::vector<std::unique_ptr<KDNode>>* kdNodes);

struct KDTree
{
    KDTree() : root(nullptr) {}
    ~KDTree() {
        delete root;
    }

    void build(std::vector<std::array<glm::vec3, 3>>& triangles)
    {
        root = new KDNode();
        root->axis = 0;

        buildTree(root, triangles, &kdNodes);
        minCorner = root->minCorner;
        maxCorner = root->maxCorner;
    }
    void clear()
    {
        delete root;
        root = nullptr;
    }

    KDNode* root;
    glm::vec3 minCorner, maxCorner; // For visualization purposes
    std::vector<std::unique_ptr<KDNode>> kdNodes;
};

