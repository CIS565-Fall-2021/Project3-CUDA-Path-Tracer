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
public:
    Scene(string filename);
    ~Scene();

    int loadTriangle(const std::array<glm::vec3, 3>& triangle, const Transform& transform, int materialId); // TODO: make private

    bool LoadObj(string filename, Transform& transform, int materialId, bool kdTree);

    std::vector<Geom> geoms;
    std::vector<KDTree> kdTrees;
    std::vector<KDNode> kdNodes;
    std::vector<Material> materials;
    RenderState state;
};
