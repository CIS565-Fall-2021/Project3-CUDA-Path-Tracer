#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "glm/glm.hpp"
#include "sceneStructs.h"
#include "static_config.h"
#include "utilities.h"

class Scene {
private:
  std::ifstream fp_in;
  int loadMaterial(std::string materialid);
  int loadGeom(std::string objectid);
  int loadCamera();

  std::vector<Geom> loadObjMesh(const std::string& file_path,
                                const std::string& material_path = "./");

public:
  Scene(std::string filename);
  ~Scene();

  std::vector<Geom> geoms;
  std::vector<Material> materials;
  RenderState state;

  struct Boundary {
    glm::vec3 min_xyz;
    glm::vec3 max_xyz;
  } boundary;
};
