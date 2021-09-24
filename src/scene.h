#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "glm/glm.hpp"
#include "sceneStructs.h"
#include "utilities.h"

using namespace std;

class Scene {
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
  RenderState state;
};
