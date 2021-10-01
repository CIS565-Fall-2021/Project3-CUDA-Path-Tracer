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
const size_t Background::COLOR_RAMP_MATERIAL_INDEX = std::numeric_limits<size_t>::max() - 1;

#define LOAD_UINT8_TEXTURE 1
#if LOAD_UINT8_TEXTURE
using STBPixelType = stbi_uc;
#define STBI_LOAD stbi_load
#else // LOAD_UINT8_TEXTURE
using STBPixelType = float;
#define STBI_LOAD stbi_loadf
#endif // LOAD_UINT8_TEXTURE

#define DEBUG_BVH_CONSTRUCTION 0//1

#if BUILD_BVH_FOR_TRIMESH
void buildBVH_CPU_NaiveRecursive(BoundingVolumeHierarchy<Triangle>& BVH, Triangle* geoms, int geomStart, int geomEnd, int rootIdx, float expand, int level = 0) {
    if (geomStart > geomEnd) {
        // No triangles.
        return;
    }
    else if (geomStart == geomEnd) {
        // Leaf node.
        BVH.nodesArray[rootIdx].box = BBox::getLocalBoundingBox(geoms[geomStart], expand);
        BVH.nodesArray[rootIdx].geomIdx = geoms[geomStart].triangleid;
        BVH.nodesArray[rootIdx].leftSubtreeIdx = -1;
        BVH.nodesArray[rootIdx].rightSubtreeIdx = -1;
#if DEBUG_BVH_CONSTRUCTION
        BBox box = BVH.nodesArray[rootIdx].box;
        printf("==Leaf: nodes[%d, %d].box={<%f,%f,%f>, <%f,%f,%f>} with geoms leaf[%d]->%d\n", 
            rootIdx, level, 
            box.minP.x, box.minP.y, box.minP.z,
            box.maxP.x, box.maxP.y, box.maxP.z,
            geomStart, geoms[geomStart].triangleid);
#endif // DEBUG_BVH_CONSTRUCTION
        return;
    }

    static struct GeomSortPredicate {
        GeomSortPredicate(i32 axis) : axis(axis) {}

        bool operator()(const Triangle& t1, const Triangle& t2) const {
            return BBox::getLocalBoundingBox(t1).getCenter()[axis] < BBox::getLocalBoundingBox(t2).getCenter()[axis];
        }

        i32 axis = 0;
    } const predicates[]{ GeomSortPredicate(0), GeomSortPredicate(1), GeomSortPredicate(2) };

    // geoms is unused after that so just sort if necessary.
    BBox box = BBox::getLocalBoundingBox(geoms[geomStart], expand);
    for (int i = geomStart + 1; i <= geomEnd; ++i) {
#if DEBUG_BVH_CONSTRUCTION > 1
        printf("--For i = %d: nodes[%d, %d].box={<%f,%f,%f>, <%f,%f,%f>} with geoms[%d:%d]\n", 
            i - 1, 
            rootIdx, level, 
            box.minP.x, box.minP.y, box.minP.z,
            box.maxP.x, box.maxP.y, box.maxP.z,
            geomStart, i - 1);
#endif // DEBUG_BVH_CONSTRUCTION
        box += BBox::getLocalBoundingBox(geoms[i], expand);
    }
#if DEBUG_BVH_CONSTRUCTION
    printf("nodes[%d, %d].box={<%f,%f,%f>, <%f,%f,%f>} with geoms[%d:%d]\n", 
        rootIdx, level, 
        box.minP.x, box.minP.y, box.minP.z,
        box.maxP.x, box.maxP.y, box.maxP.z,
        geomStart, geomEnd);
#endif // DEBUG_BVH_CONSTRUCTION

    i32 axis = box.getMaxDistAxis();
    std::sort(geoms + geomStart, geoms + geomEnd + 1, predicates[axis]);
    int geomMiddle = geomStart + ((geomEnd - geomStart) >> 1);
    int leftRoot = (rootIdx << 1) + 1;
    int rightRoot = (rootIdx << 1) + 2;

    BVH.nodesArray[rootIdx].box = box;
    BVH.nodesArray[rootIdx].geomIdx = -1;
    BVH.nodesArray[rootIdx].leftSubtreeIdx = leftRoot;
    BVH.nodesArray[rootIdx].rightSubtreeIdx = rightRoot;

    buildBVH_CPU_NaiveRecursive(BVH, geoms, geomStart, geomMiddle, leftRoot, expand, level + 1);
    buildBVH_CPU_NaiveRecursive(BVH, geoms, geomMiddle + 1, geomEnd, rightRoot, expand, level + 1);
}

template<>
__host__ void BoundingVolumeHierarchy<Triangle>::buildBVH_CPU(Triangle* geoms, int geomNum, float expand) {
    if (geomNum == 0) {
        return;
    }
    nodeNum = (geomNum << 1) - 1;
    i32 maxNodeNum = 1;
    treeHeight = 0;
    for (i32 i = 2; i - 2 < nodeNum; i <<= 1) { // 0(2)| 1(3), 2(4)| 3(5), 4(6), 5(7), 6(8)| ...
        ++treeHeight;
        maxNodeNum <<= 1;
    }
    nodeNum = (1 << (treeHeight + 1)) - 1;
    
    BoundingVolumeHierarchy<Triangle> BVHCPU(*this);

    //std::vector<BVHNode> nodesCPU(nodeNum);
    //BVHCPU.nodesArray = nodesCPU.data();
    //BVHCPU.nodesArray = new BVHNode[nodeNum];
    cudaMallocHost(&BVHCPU.nodesArray, sizeof(BVHNode) * nodeNum);
    cudaMalloc(&nodesArray, sizeof(BVHNode) * nodeNum);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < nodeNum; ++i) {
        BVHCPU.nodesArray[i] = BVHNode();
    }
    
    buildBVH_CPU_NaiveRecursive(BVHCPU, geoms, 0, geomNum - 1, 0, expand);
    
    cudaMemcpy(nodesArray, BVHCPU.nodesArray, sizeof(BVHNode) * nodeNum, cudaMemcpyHostToDevice);
    cudaFreeHost(BVHCPU.nodesArray);
    cudaDeviceSynchronize();
    //delete[] BVHCPU.nodesArray;
    printf("Initialize BVH with %d nodes, %d leaves, with height %d.\n", nodeNum, geomNum, treeHeight);
    checkCUDAError("buildBVH");
}
#endif // BUILD_BVH_FOR_TRIMESH

__global__ void kernInvGammaCorrection(glm::vec3* dst, STBPixelType* src, int x, int y, int channel) {
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

__global__ void kernGammaCorrection(Texture2D<glm::vec3> image) {
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;
    if (idxX < image.size.x && idxY < image.size.y) {
        int index = Texture2D<glm::vec3>::index2Dto1D(glm::ivec2(image.size.x, image.size.y), idxY, idxX);
        glm::vec3 srcColor = image.buffer[index];
        glm::vec3 dstColor;
        dstColor.r = powf(glm::clamp(srcColor.r, 0.f, 1.f), 1.f / 2.2f);
        dstColor.g = powf(glm::clamp(srcColor.g, 0.f, 1.f), 1.f / 2.2f);
        dstColor.b = powf(glm::clamp(srcColor.b, 0.f, 1.f), 1.f / 2.2f);
        image.buffer[index] = dstColor;
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
        kernInvGammaCorrection<<<blockCount, IMAGE_PROCESS_BLOCK_SIZE>>>(res.buffer, imageGPU, x, y, channel);
        checkCUDAError("kernInvGammaCorrection");
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
    for (auto& materialToTexturePair : texturePackage.dstToAddrMap) {
        size_t materialId = materialToTexturePair.first;
        switch (materialId) {
        case Background::BACKGROUND_MATERIAL_INDEX: {
            for (auto& textureFilePair : materialToTexturePair.second) {
                const std::string& filename = textureFilePair.second;
                Texture2D<glm::vec3>* texture_ptr = utilityCore::getPtrInStruct<Texture2D<glm::vec3>>(&background, textureFilePair.first);
                *texture_ptr = loadTexture(filename);
            }
        }
        break;
        case Background::COLOR_RAMP_MATERIAL_INDEX: {
            for (auto& textureFilePair : materialToTexturePair.second) {
                const std::string& filename = textureFilePair.second;
                Texture2D<glm::vec3>* texture_ptr = utilityCore::getPtrInStruct<Texture2D<glm::vec3>>(&rampMap, textureFilePair.first);
                *texture_ptr = loadTexture(filename);
            }
        }
        break;
        default: {
            Material& material = materials[materialId];
            for (auto& textureFilePair : materialToTexturePair.second) {
                const std::string& filename = textureFilePair.second;
                Texture2D<glm::vec3>* texture_ptr = utilityCore::getPtrInStruct<Texture2D<glm::vec3>>(&material, textureFilePair.first);
                *texture_ptr = loadTexture(filename);
            }
        }
        break;
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

            if (i0 * 3 < shape.mesh.normals.size()) {
                memcpy(&tri.nrm0, &shape.mesh.normals[i0 * 3], sizeof(tri.nrm0));
            }
            if (i1 * 3 < shape.mesh.normals.size()) {
                memcpy(&tri.nrm1, &shape.mesh.normals[i1 * 3], sizeof(tri.nrm1));
            }
            if (i2 * 3 < shape.mesh.normals.size()) {
                memcpy(&tri.nrm2, &shape.mesh.normals[i2 * 3], sizeof(tri.nrm2));
            }

            if (i0 * 2 < shape.mesh.texcoords.size()) {
                memcpy(&tri.uv00, &shape.mesh.texcoords[i0 * 2], sizeof(tri.uv00));
            }
            if (i1 * 2 < shape.mesh.texcoords.size()) {
                memcpy(&tri.uv01, &shape.mesh.texcoords[i1 * 2], sizeof(tri.uv01));
            }
            if (i2 * 2 < shape.mesh.texcoords.size()) {
                memcpy(&tri.uv02, &shape.mesh.texcoords[i2 * 2], sizeof(tri.uv02));
            }
            
            ++triIdx;
        }
    }
    cudaMalloc(&res.triangles, sizeof(Triangle) * res.triangleNum);
    cudaMemcpy(res.triangles, tris.data(), sizeof(Triangle) * res.triangleNum, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    checkCUDAError("loadModelObj");
#if BUILD_BVH_FOR_TRIMESH
    res.localBVH.buildBVH_CPU(tris.data(), tris.size(), FLT_EPSILON);
#endif // BUILD_BVH_FOR_TRIMESH
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

void Scene::initGBuffer() {
    dev_frameBuffer.size = state.camera.resolution;
    cudaMalloc(&dev_frameBuffer.buffer, sizeof(glm::vec3) * dev_frameBuffer.size.x * dev_frameBuffer.size.y);

    dev_GBuffer.size = state.camera.resolution;
    cudaMalloc(&dev_GBuffer.buffer, sizeof(GBufferData) * dev_GBuffer.size.x * dev_GBuffer.size.y);
    cudaDeviceSynchronize();
    std::cout << "Initialized frame buffer " << dev_frameBuffer.size.x << " x " << dev_frameBuffer.size.y << std::endl;
    std::cout << "Initialized G-buffer " << dev_GBuffer.size.x << " x " << dev_GBuffer.size.y << std::endl;
    checkCUDAError("cudaMalloc GBuffer");
}

void Scene::freeGBuffer() {
    cudaFree(dev_GBuffer.buffer);

    cudaFree(dev_frameBuffer.buffer);
    cudaDeviceSynchronize();
    checkCUDAError("cudaFree GBuffer");
}

namespace PostProcessGPU {
#if !PREGATHER_FINAL_IMAGE
    __global__ void dividedByIter(Texture2D<glm::vec3> dst, Texture2D<glm::vec3> src, float iter) {
        int idxX = blockDim.x * blockIdx.x + threadIdx.x;
        int idxY = blockDim.y * blockIdx.y + threadIdx.y;
        if (idxX < dst.size.x && idxY < dst.size.y) {
            dst.setPixelByHW(idxY, idxX, src.getPixelByHW(idxY, idxX) / iter);
        }
    }
#endif // PREGATHER_FINAL_IMAGE
    extern __global__ void postProcess_ColorRamp(
        Texture2D<glm::vec3> imageTexture, 
        Texture2D<GBufferData> gBuffer, 
        Texture2D<glm::vec3> rampTexture);

    extern __global__ void postProcess_OutlineByStencil(
        Texture2D<glm::vec3> imageTexture, 
        Texture2D<GBufferData> gBuffer, 
        int stencilId, glm::vec3 outlineColor, int outlineWidth);
}


glm::vec3* Scene::postProcessGPU(glm::vec3* dev_image, PathSegment* dev_paths, const dim3 blocksPerGrid2d, const dim3 blockSize2d, int iter) const {
    Texture2D<glm::vec3> imageTexture;
    imageTexture.buffer = dev_image;
    imageTexture.size = dev_GBuffer.size;
#if !PREGATHER_FINAL_IMAGE
    PostProcessGPU::dividedByIter<<<blocksPerGrid2d, blockSize2d>>>(dev_frameBuffer, imageTexture, iter);
#else // PREGATHER_FINAL_IMAGE
    cudaMemcpy(dev_frameBuffer.buffer, dev_image, sizeof(glm::vec3) * dev_GBuffer.size.x * dev_GBuffer.size.y, cudaMemcpyDeviceToDevice);
#endif // PREGATHER_FINAL_IMAGE
    for (size_t i = 0; i < postprocesses.size(); ++i) {
        auto& pppair = postprocesses[i];
        if (!pppair.second) {
            continue;
        }
        PostProcessType pptype = pppair.first;
        switch (pptype) {
        case PostProcessType::COLOR_RAMP:
            if (rampMap.isReadable()) {
                PostProcessGPU::postProcess_ColorRamp<<<blocksPerGrid2d, blockSize2d>>>(dev_frameBuffer, dev_GBuffer, rampMap);
            }
            break;
        case PostProcessType::OUTLINE_BY_STENCIL:
        {
            int stencilId = ppToStencilMap.at(i);
            auto colorWidthPair = stencilOutlineColorWidths.at(stencilId);
            //printf("%d<%f,%f,%f>%d\n", stencilId, colorWidthPair.first.r, colorWidthPair.first.g, colorWidthPair.first.b, colorWidthPair.second);
            PostProcessGPU::postProcess_OutlineByStencil<<<blocksPerGrid2d, blockSize2d>>>(dev_frameBuffer, dev_GBuffer, stencilId, colorWidthPair.first, colorWidthPair.second);
        }
            break;
        }
        checkCUDAError(("postprocess" + std::to_string(i)).c_str());
    }
    kernGammaCorrection<<<blocksPerGrid2d, blockSize2d>>>(dev_frameBuffer);
    checkCUDAError("gamma correction");
    return dev_frameBuffer.buffer;
}

__global__ void PostProcessGPU::postProcess_ColorRamp(Texture2D<glm::vec3> imageTexture, Texture2D<GBufferData> gBuffer, Texture2D<glm::vec3> rampTexture) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;
    if (idxX < imageTexture.size.x && idxY < imageTexture.size.y) {
        //GBufferData gBufferData = gBuffer.getPixelByHW(idxY, idxX);
        glm::vec3 color = imageTexture.getPixelByHW(idxY, idxX);
        color = glm::clamp(color, 0.f, 1.f);
        glm::vec3 ramp;
        ramp.r = rampTexture.getPixelByUV(color.r, 0.5f).r;
        ramp.g = rampTexture.getPixelByUV(color.g, 0.5f).g;
        ramp.b = rampTexture.getPixelByUV(color.b, 0.5f).b;
        //printf("ramp of <%f,%f,%f> is <%f,%f,%f>\n", color.r, color.g, color.b, ramp.r, ramp.g, ramp.b);

        int index = Texture2D<glm::vec3>::index2Dto1D(imageTexture.size, idxY, idxX);
        imageTexture.buffer[index] = ramp;//glm::clamp(ramp, 0.f, 1.f);
    }
}

__global__ void PostProcessGPU::postProcess_OutlineByStencil(Texture2D<glm::vec3> imageTexture, Texture2D<GBufferData> gBuffer, int stencilId, glm::vec3 outlineColor, int outlineWidth) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;
    if (idxX < imageTexture.size.x && idxY < imageTexture.size.y) {
        GBufferData data = gBuffer.getPixelByHW(idxY, idxX);
        if (data.stencilId == stencilId) {
            return;
        }
        //printf("%d,%d stencil = %d\n", idxX, idxY, data.stencilId);
        for (int y = glm::max(0, idxY - outlineWidth); y <= glm::min(imageTexture.size.y - 1, idxY + outlineWidth); ++y) {
            for (int x = glm::max(0, idxX - outlineWidth); x <= glm::min(imageTexture.size.x - 1, idxX + outlineWidth); ++x) {
                GBufferData data1 = gBuffer.getPixelByHW(y, x);
                if (data1.stencilId == stencilId) {
                    imageTexture.setPixelByHW(idxY, idxX, outlineColor);
                    return;
                }
            }
        }
    }
}
