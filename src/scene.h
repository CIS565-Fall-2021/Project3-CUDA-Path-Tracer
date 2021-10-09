#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    void Scene::loadNode(tinygltf::Model& model, tinygltf::Node& node);
    void Scene::loadMesh(tinygltf::Model& model, tinygltf::Mesh& mesh);
public:
    Scene(string filename);
    void Scene::addGltf(tinygltf::Model& model);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<Mesh> meshes;
    RenderState state;
};
