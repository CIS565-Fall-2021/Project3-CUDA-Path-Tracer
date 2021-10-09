#pragma once

#include <vector>
#include <array>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadTriangle(Triangle& triangle, const Transform& transform, int materialId);
    Geom createTriangle(Triangle& triangle, const Transform& transform, int materialId);

public:
    Scene(string filename);
    ~Scene();


    bool LoadObj(string filename, Transform& transform, int materialId, bool bvhTree);

    std::vector<Geom> geoms;
    std::vector<Triangle> primitives;
    std::vector<BVHTree> bvhTrees;
    std::vector<BVHNode> bvhNodes;
    std::vector<Material> materials;
    RenderState state;
};
