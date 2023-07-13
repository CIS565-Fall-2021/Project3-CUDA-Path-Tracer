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
    int loadGLTFNode(const std::vector<tinygltf::Node>& nodes, 
      const tinygltf::Node& node, const glm::mat4& xform, bool* isLoaded);
    int loadGLTF(const std::string& filename, float scale);
public:
    Scene(string filename);
    ~Scene() {};

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;

    std::vector<Mesh>      meshes;
    std::vector<Primitive> primitives;

    // Primitive Data
    std::vector<uint16_t>  mesh_indices;
    std::vector<glm::vec3> mesh_vertices;
    std::vector<glm::vec3> mesh_normals;
    std::vector<glm::vec2> mesh_uvs;
    std::vector<glm::vec4> mesh_tangents;

    RenderState state;
};
