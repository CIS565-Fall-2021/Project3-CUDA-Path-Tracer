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
    void load_obj(string path, Geom &geom, int offset);
    void load_texture(const char *path, Geom &geom, int offset, bool is_normal_map);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> meshes;
    std::vector<glm::vec3> textures;
    std::vector<glm::vec3> normal_maps;
    int mesh_offset;
    int texture_offset;
    int normal_offset;
    RenderState state;
};
