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

struct Transform
{
    glm::vec3 translate;
    glm::vec3 rotate;
    glm::vec3 scale;
};

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    Geom createTriangle(const std::array<glm::vec3, 3>& triangle, const Transform& transform, int materialId);
public:
    Scene(string filename);
    ~Scene();

    int loadTriangle(const std::array<glm::vec3, 3>& triangle, const Transform& transform, int materialId); // TODO: make private

    bool LoadObj(string filename, Transform transform, int materialId, bool kdTree);

    std::vector<Geom> geoms;
    std::vector<std::unique_ptr<KDTree>> kdTrees;
    std::vector<Material> materials;
    RenderState state;
};
