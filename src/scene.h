#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "sceneStructs.h"


class Scene {
private:
	std::ifstream fp_in;
	int loadMaterial(std::string materialid);
	int loadGeom(std::string objectid);
	int loadCamera();
public:
	Scene(std::string filename);

	std::vector<Geom> geoms;
	std::vector<Material> materials;
	RenderState state;
};
