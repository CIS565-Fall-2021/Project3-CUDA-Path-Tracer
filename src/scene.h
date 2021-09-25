#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

class Scene {
private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    int loadGeom(std::string objectid);
    int loadCamera();
protected:
    virtual bool readFromToken(const std::vector<std::string>& tokens);
public:
    Scene(std::string filename);
    virtual ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
