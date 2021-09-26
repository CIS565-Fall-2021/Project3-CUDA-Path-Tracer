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
#include "scene.h"

const dim3 IMAGE_PROCESS_BLOCK_SIZE(16, 16, 1);

__global__ void invGammaCorrect(glm::vec3* dst, stbi_uc* src, int x, int y, int channel) {
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;
    if (idxX < x && idxY < y) {
        int index = idxY * x + idxX;
        glm::vec3 color;
#pragma unroll
        for (int c = 0; c < channel && c < 3; ++c) {
            stbi_uc byte = src[index * channel + c];
            float comp = byte / 255.f;
            comp = powf(comp, 2.2);
            color[c] = comp > 1.f ? 1.f : (comp < 0.f ? 0.f : comp);
        }
        dst[index] = color;
    }
}


Texture2D<glm::vec3> Scene::loadTexture(const std::string& filename) {
    auto it = textureIdMap.find(filename);
    if (it != textureIdMap.end()) {
        return textures[it->second];
    }

    int x, y, channel;
    stbi_uc* imageCPU = stbi_load(filename.c_str(), &x, &y, &channel, 0);
    stbi_uc* imageGPU;
    textureIdMap[filename] = textures.size();
    textures.emplace_back();
    Texture2D<glm::vec3>& tex = textures.back();
    
    tex.size.x = x;
    tex.size.y = y;

    cudaMalloc(&imageGPU, sizeof(stbi_uc) * x * y * channel);
    cudaMalloc(&tex.buffer, sizeof(glm::vec3) * x * y);

    cudaMemcpy(imageGPU, imageCPU, sizeof(stbi_uc) * x * y * channel, cudaMemcpyHostToDevice);
    dim3 blockCount((x + IMAGE_PROCESS_BLOCK_SIZE.x - 1) / IMAGE_PROCESS_BLOCK_SIZE.x, (y + IMAGE_PROCESS_BLOCK_SIZE.y - 1) / IMAGE_PROCESS_BLOCK_SIZE.y, 1);
    invGammaCorrect<<<blockCount, IMAGE_PROCESS_BLOCK_SIZE>>>(tex.buffer, imageGPU, x, y, channel);

    cudaDeviceSynchronize();
    stbi_image_free(imageCPU);
    return tex;
}

void Scene::freeTextures() {
    for (Texture2D<glm::vec3>& tex : textures) {
        cudaFree(tex.buffer);
    }
    cudaDeviceSynchronize();
}