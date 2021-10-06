#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

// Hardcoded for now
#define TEXWIDTH 4096
#define TEXHEIGHT 4096

class Scene
{
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<struct Triangle> triangles;
    int numMeshs = 0; // number of start indices
    RenderState state;

    std::vector<struct TexData> texData;
};
