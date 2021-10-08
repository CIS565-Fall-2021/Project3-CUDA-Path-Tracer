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


struct KDNode
{
    KDNode() : leftChild(-1), rightChild(-1), axis(0), particles() {}
    ~KDNode() {}

    int leftChild; 
    int rightChild; 
    unsigned int axis; // Which axis split this node represents
    Geom boundingBox;
    Geom triangle; // only initialized if this is a leaf node
    std::vector<std::array<glm::vec3, 3>> particles; // A collection of pointers to the particles contained in this node.
};

// TODO: use getMinVertex to clean up code
bool xSort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b);
bool ySort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b);
bool zSort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b);

void buildTree(
    int node, 
    std::vector<std::array<glm::vec3, 3>>& triangles, 
    std::vector<KDNode>* kdNodes);

Geom createTriangle(const std::array<glm::vec3, 3>& triangle, int materialId);
Geom createTriangle(const std::array<glm::vec3, 3>& triangle, const Transform& transform, int materialId);

struct KDTree
{
    KDTree(int inx) : kdNodes(inx)
    {}
    ~KDTree() {}

    void updateLeafNodes(std::vector<KDNode>* kdNodes, int materialId)
    {
        for (auto& kdNode : *kdNodes)
        {
            if (kdNode.leftChild == -1 && kdNode.rightChild == -1)
            {
                if (kdNode.particles.size() != 0) // TODO: why does this sometimes NOT happen?
                    kdNode.triangle = createTriangle(kdNode.particles.at(0), materialId);
            }
        }
    }

    void build(std::vector<KDNode>* kdNodes, std::vector<std::array<glm::vec3, 3>>& triangles, int materialId)
    {
        KDNode root = KDNode();
        root.axis = 0;
        kdNodes->push_back(root);
        buildTree(0, triangles, kdNodes);
        updateLeafNodes(kdNodes, materialId);
    }
    int kdNodes;
};

