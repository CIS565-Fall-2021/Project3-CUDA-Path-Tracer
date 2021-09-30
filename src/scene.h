#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "glm/glm.hpp"
#include "sceneStructs.h"
#include "utilities.h"

class Scene {
private:
  std::ifstream fp_in;
  int loadMaterial(std::string materialid);
  int loadGeom(std::string objectid);
  int loadCamera();

public:
  Scene(std::string filename);
  ~Scene();

  std::vector<Geom> geoms;
  std::vector<Material> materials;
  RenderState state;
};
