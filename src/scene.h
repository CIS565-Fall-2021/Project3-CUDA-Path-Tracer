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
	int load_obj(std::string inputfile, glm::vec3 &mincoords, glm::vec3 &maxcoords);
public:
	Scene(std::string filename);

	std::vector<Geom> geoms;
	std::vector<Material> materials;
	std::vector<Triangle> tris;
	RenderState state;
};
