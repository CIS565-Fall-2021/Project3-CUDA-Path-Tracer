#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <queue>
#include <functional>
#include <unordered_map>
#include <stb_image.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

extern void checkCUDAErrorFn(const char* msg, const char* file, int line);

class Scene {
private:
    int loadMaterial(std::string materialid);
    int loadGeom(std::string objectid);
    int loadCamera();
    int loadBackground();

    std::ifstream fp_in;
    
protected:
    virtual bool readFromToken(const std::vector<std::string>& tokens);

    template<typename TPak>
    struct LoadingPackage {
        std::unordered_map<std::string, size_t> idMap;
        std::unordered_map<size_t, std::unordered_map<i64, std::string>> dstToAddrMap;
        std::vector<TPak> resources;
    };

    void addTextureToLoad(size_t id, i64 varOffset, const std::string& filename);

    Texture2D<glm::vec3> loadTexture(const std::string& filename);
    void initTextures();
    void freeTextures();

    LoadingPackage<Texture2D<glm::vec3>> texturePackage;

    void addModelToLoad(size_t id, i64 varOffset, const std::string& filename);

    TriMesh loadModelObj(const std::string& filename);
    void initModels();
    void freeModels();

    LoadingPackage<TriMesh> modelPackage;

    //std::unordered_map<std::string, size_t> textureIdMap;
    //std::unordered_map<size_t, std::unordered_map<i64, std::string>> materialToTextureMap;
    //std::vector<Texture2D<glm::vec3>> textureBuffers;

    std::string basePath;
    std::queue<std::function<void(void)>> initCallbacks;
public:
    Scene(std::string filename);
    virtual ~Scene();
    void execInitCallbacks();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    glm::vec3 backgroundColor;
};
