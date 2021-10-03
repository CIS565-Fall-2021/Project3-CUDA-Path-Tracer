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
public:
    Scene(string filename);
    ~Scene();

    int loadTriangle(const std::array<glm::vec3, 3>& triangle, const Transform& transform, int materialId);

    bool LoadObj(string filename, Transform transform, int materialId);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
