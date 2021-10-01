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

enum class PostProcessType {
    NONE,
    COLOR_RAMP,
    OUTLINE_BY_STENCIL,
};

extern void checkCUDAErrorFn(const char* msg, const char* file, int line);

struct Background {
    GLM_FUNC_QUALIFIER glm::vec3 getBackgroundColor(glm::vec3 dir) const {
        return sphereMap.isReadable() ? sphereMap.getPixelFromSphereMap(dir) : backgroundColor;
    }

    Texture2D<glm::vec3> sphereMap;
    glm::vec3 backgroundColor;

    static const size_t BACKGROUND_MATERIAL_INDEX;
    static const size_t COLOR_RAMP_MATERIAL_INDEX;
};

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

    void initGBuffer();
    void freeGBuffer();

    Texture2D<glm::vec3> rampMap;
    std::unordered_map<int, int> ppToStencilMap;
    std::unordered_map<int, std::pair<glm::vec3, int>> stencilOutlineColorWidths;
public:
    Scene(std::string filename);
    virtual ~Scene();
    void execInitCallbacks();
    glm::vec3* postProcessGPU(glm::vec3* dev_image, PathSegment* dev_paths, const dim3 blocksPerGrid2d, const dim3 blockSize2d, int iter) const;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    Background background;

    Texture2D<glm::vec3> dev_frameBuffer;
    Texture2D<GBufferData> dev_GBuffer;
    std::vector<std::pair<PostProcessType, bool>> postprocesses;
};
