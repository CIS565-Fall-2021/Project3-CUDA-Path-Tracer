#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#define MESH_CULL 1

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadMesh(string objectid, unsigned int meshId);
#if MESH_CULL
    int triangleIndex;
#endif

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
#if MESH_CULL
    std::vector<Mesh> meshes;
    std::vector<Geom> triangles;
#endif
};
