#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <stb_image.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

class Scene {
private:
    std::ifstream fp_in;
    
    std::unordered_map<std::string, int> textureIdMap;
    std::vector<Texture2D<glm::vec3>> textures;
    
    int loadMaterial(std::string materialid);
    int loadGeom(std::string objectid);
    int loadCamera();
    int loadBackground();
protected:
    Texture2D<glm::vec3> loadTexture(const std::string& filename);
    virtual bool readFromToken(const std::vector<std::string>& tokens);
    void freeTextures();
public:
    Scene(std::string filename);
    virtual ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    glm::vec3 backgroundColor;
};
