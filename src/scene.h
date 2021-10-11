#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadCameraGLTF(tinygltf::Model);
    int loadMaterialOBJ(tinyobj::ObjReader reader);
    int loadCameraOBJ(tinyobj::ObjReader reader);
    int loadGeomOBJ(tinyobj::ObjReader reader);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    glm::vec3 obj_mins;
    glm::vec3 obj_maxs;

    std::vector<Triangle> triangles;


};
