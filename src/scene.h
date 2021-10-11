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
    int loadGLTF(string filename, Geom& geomTemplate);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    vector<Geom> geoms;
    vector<Triangle> triangles;
    vector<Material> materials;
    vector<glm::vec3> texData;
    RenderState state;
};
