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

__host__ __device__ Geom createTriangle(Triangle& triangle, int materialId);

__host__ __device__ Geom createTriangle(Triangle& triangle, const Transform& transform, int materialId);

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    int loadTriangle(Triangle& triangle, const Transform& transform, int materialId); // TODO: make private

    bool LoadObj(string filename, Transform& transform, int materialId, bool kdTree);

    std::vector<Geom> geoms;
    std::vector<Triangle> primitives;
    std::vector<KDTree> kdTrees;
    std::vector<KDNode> kdNodes;
    std::vector<Material> materials;
    RenderState state;
};
