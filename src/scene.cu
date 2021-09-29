#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <stb_image.h>
#include "thirdparty/tiny_obj_loader.h"
#include "scene.h"

const dim3 IMAGE_PROCESS_BLOCK_SIZE(16, 16, 1);
const size_t Background::BACKGROUND_MATERIAL_INDEX = std::numeric_limits<size_t>::max();

#define LOAD_UINT8_TEXTURE 1
#if LOAD_UINT8_TEXTURE
using STBPixelType = stbi_uc;
#define STBI_LOAD stbi_load
#else // LOAD_UINT8_TEXTURE
using STBPixelType = float;
#define STBI_LOAD stbi_loadf
#endif // LOAD_UINT8_TEXTURE

__global__ void kernInvGammaCorrect(glm::vec3* dst, STBPixelType* src, int x, int y, int channel) {
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;
    if (idxX < x && idxY < y) {
        int index = Texture2D<glm::vec3>::index2Dto1D(glm::ivec2(x, y), idxY, idxX);
        glm::vec3 color;
#pragma unroll
        for (int c = 0; c < channel && c < 3; ++c) {
#if LOAD_UINT8_TEXTURE
            float comp = glm::clamp(src[index * channel + c] / 255.f, 0.f, 1.f);
#else // LOAD_UINT8_TEXTURE
            float comp = glm::clamp(src[index * channel + c], 0.f, 1.f);
#endif // LOAD_UINT8_TEXTURE
            comp = powf(comp, 2.2);
            color[c] = glm::clamp(comp, 0.f, 1.f);
        }
        dst[index] = color;
    }
}


void Scene::addTextureToLoad(size_t id, i64 varOffset, const std::string& filename) {
    auto& pkg = texturePackage;
    auto it = pkg.dstToAddrMap.find(id);
    std::unordered_map<i64, std::string>* map_ptr = nullptr;
    if (it != pkg.dstToAddrMap.end()) {
        map_ptr = &it->second;
    }
    else {
        pkg.dstToAddrMap[id] = std::unordered_map<i64, std::string>();
        map_ptr = &pkg.dstToAddrMap[id];
    }
    (*map_ptr)[varOffset] = filename;
}

Texture2D<glm::vec3> Scene::loadTexture(const std::string& filename) {
    std::cout << "Loading Texture from " << filename << "..." << std::endl;
    auto& pkg = texturePackage;
    auto it = pkg.idMap.find(filename);
    if (it != pkg.idMap.end()) {
        std::cout << "Texture " << filename << " is already exist.\n" << std::endl;
        return pkg.resources[it->second];
    }

    pkg.idMap[filename] = pkg.resources.size();
    pkg.resources.emplace_back();
    auto& res = pkg.resources.back();

    //Texture2D<glm::vec3>& tex = textureBuffers[i];
    int x, y, channel; 
    stbi_set_flip_vertically_on_load(1);

    std::string extension = utilityCore::getFileExtension(filename);
    if (stricmp(extension.c_str(), "hdr") == 0) {
        float* imageCPU = stbi_loadf(filename.c_str(), &x, &y, &channel, 0);
        if (!imageCPU) {
            std::cout << "Texture " << filename << ": failed to load.\n" << std::endl;
        }
        cudaMalloc(&res.buffer, sizeof(glm::vec3) * x * y);
        cudaMemcpy(res.buffer, imageCPU, sizeof(float) * x * y * channel, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        stbi_image_free(imageCPU);
    }
    else {
        STBPixelType* imageCPU = STBI_LOAD(filename.c_str(), &x, &y, &channel, 0);
        if (!imageCPU) {
            std::cout << "Texture " << filename << ": failed to load.\n" << std::endl;
        }
        cudaMalloc(&res.buffer, sizeof(glm::vec3) * x * y);

        STBPixelType* imageGPU;
        cudaMalloc(&imageGPU, sizeof(STBPixelType) * x * y * channel);
        cudaMemcpy(imageGPU, imageCPU, sizeof(STBPixelType) * x * y * channel, cudaMemcpyHostToDevice);
        dim3 blockCount((x + IMAGE_PROCESS_BLOCK_SIZE.x - 1) / IMAGE_PROCESS_BLOCK_SIZE.x, (y + IMAGE_PROCESS_BLOCK_SIZE.y - 1) / IMAGE_PROCESS_BLOCK_SIZE.y, 1);
        kernInvGammaCorrect<<<blockCount, IMAGE_PROCESS_BLOCK_SIZE>>>(res.buffer, imageGPU, x, y, channel);
        checkCUDAError("kernInvGammaCorrect");
        cudaFree(imageGPU);
        cudaDeviceSynchronize();
        stbi_image_free(imageCPU);
    }

    res.size.x = x;
    res.size.y = y;

    checkCUDAError("cudaFree imageGPU");
    std::cout << "Texture " << filename << '<' << x << ',' << y << ',' << channel << "> created.\n" << std::endl;

    return res;
}

void Scene::initTextures() {
    for(auto& materialToTexturePair : texturePackage.dstToAddrMap) {
        size_t materialId = materialToTexturePair.first;
        if (materialId != Background::BACKGROUND_MATERIAL_INDEX) {
            Material& material = materials[materialId];
            for (auto& textureFilePair : materialToTexturePair.second) {
                const std::string& filename = textureFilePair.second;
                Texture2D<glm::vec3>* texture_ptr = utilityCore::getPtrInStruct<Texture2D<glm::vec3>>(&material, textureFilePair.first);
                *texture_ptr = loadTexture(filename);
            }
        }
        else { // Background texture
            for (auto& textureFilePair : materialToTexturePair.second) {
                const std::string& filename = textureFilePair.second;
                Texture2D<glm::vec3>* texture_ptr = utilityCore::getPtrInStruct<Texture2D<glm::vec3>>(&background, textureFilePair.first);
                *texture_ptr = loadTexture(filename);
            }
        }
    }
}

void Scene::freeTextures() {
    for (auto& tex : texturePackage.resources) {
        cudaFree(tex.buffer);
    }
    cudaDeviceSynchronize();
    checkCUDAError("cudaFree textures");
}

void Scene::addModelToLoad(size_t id, i64 varOffset, const std::string& filename) {
    auto& pkg = modelPackage;
    auto it = pkg.dstToAddrMap.find(id);
    std::unordered_map<i64, std::string>* map_ptr = nullptr;
    if (it != pkg.dstToAddrMap.end()) {
        map_ptr = &it->second;
    }
    else {
        pkg.dstToAddrMap[id] = std::unordered_map<i64, std::string>();
        map_ptr = &pkg.dstToAddrMap[id];
    }
    (*map_ptr)[varOffset] = filename;
}

TriMesh Scene::loadModelObj(const std::string& filename) {
    std::cout << "Loading Model from " << filename << "..." << std::endl;
    auto& pkg = modelPackage;
    auto it = pkg.idMap.find(filename);
    if (it != pkg.idMap.end()) {
        std::cout << "Model " << filename << " is already exist.\n" << std::endl;
        return pkg.resources[it->second];
    }

    pkg.idMap[filename] = pkg.resources.size();
    pkg.resources.emplace_back();
    auto& res = pkg.resources.back();

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    tinyobj::LoadObj(shapes, materials, filename.c_str());

    res.triangleNum = 0;
    for (tinyobj::shape_t& shape : shapes) {
        res.triangleNum += shape.mesh.indices.size() / 3;
    }

    std::vector<Triangle> tris(res.triangleNum);
    size_t triIdx = 0;
    for (tinyobj::shape_t& shape : shapes) {
        for (int i = 0; i < shape.mesh.indices.size(); i += 3) {
            auto& tri = tris[triIdx];
            tri.triangleid = triIdx;
            tri.twoSided = true;

            //memcpy(tri.position, &shape.mesh.positions[i0 * 3], sizeof(tri.position));
            //memcpy(tri.normal, &shape.mesh.normals[i0 * 3], sizeof(tri.normal));
            //memcpy(tri.uv0, &shape.mesh.texcoords[i0 * 2], sizeof(tri.uv0));

            size_t i0 = shape.mesh.indices[i];
            size_t i1 = shape.mesh.indices[i + 1];
            size_t i2 = shape.mesh.indices[i + 2];

            memcpy(&tri.pos0, &shape.mesh.positions[i0 * 3], sizeof(tri.pos0));
            memcpy(&tri.pos1, &shape.mesh.positions[i1 * 3], sizeof(tri.pos1));
            memcpy(&tri.pos2, &shape.mesh.positions[i2 * 3], sizeof(tri.pos2));

            memcpy(&tri.nrm0, &shape.mesh.normals[i0 * 3], sizeof(tri.nrm0));
            memcpy(&tri.nrm1, &shape.mesh.normals[i1 * 3], sizeof(tri.nrm1));
            memcpy(&tri.nrm2, &shape.mesh.normals[i2 * 3], sizeof(tri.nrm2));

            memcpy(&tri.uv00, &shape.mesh.texcoords[i0 * 2], sizeof(tri.uv00));
            memcpy(&tri.uv01, &shape.mesh.texcoords[i1 * 2], sizeof(tri.uv01));
            memcpy(&tri.uv02, &shape.mesh.texcoords[i2 * 2], sizeof(tri.uv02));
            
            ++triIdx;
        }
    }
    cudaMalloc(&res.triangles, sizeof(Triangle) * res.triangleNum);
    cudaMemcpy(res.triangles, tris.data(), sizeof(Triangle) * res.triangleNum, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    checkCUDAError("loadModelObj");
    std::cout << "Model " << filename << '<' << res.triangleNum << "> created.\n" << std::endl;
    return res;
}

void Scene::initModels() {
    for(auto& geomToModelPair : modelPackage.dstToAddrMap) {
        size_t geomId = geomToModelPair.first;
        Geom& geom = geoms[geomId];
        for (auto& modelFilePair : geomToModelPair.second) {
            const std::string& filename = modelFilePair.second;
            TriMesh* model_ptr = utilityCore::getPtrInStruct<TriMesh>(&geom, modelFilePair.first);
            std::string extension = utilityCore::getFileExtension(filename);
            if (stricmp("obj", extension.c_str()) == 0) {
                *model_ptr = loadModelObj(filename);
            }
            //TODO: Other model format.
        }
        //TODO: Build BVH if necessary.
    }
}

void Scene::freeModels() {
    for (auto& mdl : modelPackage.resources) {
        cudaFree(mdl.triangles);
    }
    cudaDeviceSynchronize();
    checkCUDAError("cudaFree models");
}
