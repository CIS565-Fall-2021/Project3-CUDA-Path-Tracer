#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;
#define MESH_BOUND_CHECK 0


class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadObjFile();
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();
   
    std::vector<TriangleGeom> triangles;

#ifdef MESH_BOUND_CHECK
    glm::vec3 triangle_bound_min;
    glm::vec3 triangle_bound_max;
#endif

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
