#pragma once

#include <vector>
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
    bool loadObj(string filename, Geom& geom);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    // just need scene to own the unique ptrs
    std::vector<unique_ptr<std::vector<Triangle>>> trianglePtrs;
   // std::vector<unique_ptr<vector<glm::vec3>>> trianglePtrs;
    //std::vector<glm::vec3> trianglesTest;
    RenderState state;
};
