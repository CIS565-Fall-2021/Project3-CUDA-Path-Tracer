#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadBackground();

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<struct Triangle> triangles;
    int numMeshs = 0; // number of start indices
    RenderState state;

    std::vector<struct TexData> texData;
    int backWidth = 0;
    int backHeight = 0;
    std::vector<glm::vec3> backTex;
};
