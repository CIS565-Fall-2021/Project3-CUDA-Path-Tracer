#pragma once

#include "glm/glm.hpp"
#include "../utilities.h"

template<typename TPixelType>
struct Texture2D {
    //GLM_FUNC_QUALIFIER operator bool() const;
    GLM_FUNC_QUALIFIER bool isReadable() const;
    GLM_FUNC_QUALIFIER void setPixelByHW(int h, int w, TPixelType val);
    GLM_FUNC_QUALIFIER TPixelType getPixelByHW(int h, int w) const;
    GLM_FUNC_QUALIFIER TPixelType getPixelByUV(float u, float v) const;
    GLM_FUNC_QUALIFIER TPixelType getPixelByUVBilinear(float u, float v) const;

    GLM_FUNC_QUALIFIER TPixelType getPixelFromSphereMap(glm::vec3 dir) const;

    static GLM_FUNC_QUALIFIER i32 index2Dto1D(glm::ivec2 size, i32 h, i32 w) {
        return h * size.x + w;
    }

    TPixelType* buffer = nullptr;
    glm::ivec2 size;
};

//template<typename TPixelType>
//GLM_FUNC_QUALIFIER Texture2D<TPixelType>::operator bool() const {
//    return isReadable();
//}

template<typename TPixelType>
GLM_FUNC_QUALIFIER bool Texture2D<TPixelType>::isReadable() const {
    return buffer != nullptr;
}

template<typename TPixelType>
GLM_FUNC_QUALIFIER void Texture2D<TPixelType>::setPixelByHW(int h, int w, TPixelType val) {
    //printf("Write texture<%d,%d>[%d,%d]\n", size.y, size.x, h, w);
    buffer[index2Dto1D(size, h, w)] = val;
}

template<typename TPixelType>
GLM_FUNC_QUALIFIER TPixelType Texture2D<TPixelType>::getPixelByHW(int h, int w) const {
    //printf("Read texture<%d,%d>[%d,%d]\n", size.y, size.x, h, w);
    return buffer[index2Dto1D(size, h, w)];
}

template<typename TPixelType>
GLM_FUNC_QUALIFIER TPixelType Texture2D<TPixelType>::getPixelByUV(float u, float v) const {
    int width = size.x, height = size.y;
    int u_img = glm::clamp<i32>(u * width + 0.5f, 0, width - 1);
    int v_img = glm::clamp<i32>(v * height + 0.5f, 0, height - 1);
    return getPixelByHW(v_img, u_img);
}

template<typename TPixelType>
GLM_FUNC_QUALIFIER TPixelType Texture2D<TPixelType>::getPixelByUVBilinear(float u, float v) const {
    int width = size.x, height = size.y;
    float u_img = glm::clamp(u, 0.f, 1.f) * width;
    float v_img = glm::clamp(v, 0.f, 1.f) * height;
    //float v_img = (1 - v) * height;

    float u_img_f = floor(u_img);//glm::max(0.f, floor(u_img));
    float u_img_c = ceil(u_img);//glm::min(static_cast<float>(width - 1), ceil(u_img));
    float v_img_f = floor(v_img);//glm::max(0.f, floor(v_img));
    float v_img_c = ceil(u_img);//glm::min(static_cast<float>(height - 1), ceil(v_img));

    int ufi = glm::max<i32>(0, u_img_f + 0.5f);
    int uci = glm::min<i32>(width - 1,u_img_c + 0.5f);
    int vfi = glm::max<i32>(0, v_img_f + 0.5f);
    int vci = glm::min<i32>(height - 1, v_img_c + 0.5f);

    auto color00 = getPixelByHW(vfi, ufi);
    auto color01 = getPixelByHW(vci, ufi);
    auto color10 = getPixelByHW(vfi, uci);
    auto color11 = getPixelByHW(vci, uci);

    float dU = u_img_c - u_img_f;
    float dV = v_img_c - v_img_f;

    float wuf = ufi == uci ? 0.5f : (u_img - u_img_f) / dU;
    float wuc = ufi == uci ? 0.5f : (u_img_c - u_img) / dU;
    float wvf = vfi == vci ? 0.5f : (v_img - v_img_f) / dV;
    float wvc = vfi == vci ? 0.5f : (v_img_c - v_img) / dV;

    float w11 = wuf * wvf;
    float w10 = wuf * wvc;
    float w01 = wuc * wvf;
    float w00 = wuc * wvc;

    auto color00f = 
        TPixelType(color00[0], color00[1], color00[2]);
    auto color01f = 
        TPixelType(color01[0], color01[1], color01[2]);
    auto color10f = 
        TPixelType(color10[0], color10[1], color10[2]);
    auto color11f = 
        TPixelType(color11[0], color11[1], color11[2]);

    auto color = 
        color00f * w00 +
        color01f * w01 +
        color10f * w10 +
        color11f * w11;
    return color;
}

#define SPHERE_MAP_FACTOR_U HALF_INV_PI
#define SPHERE_MAP_FACTOR_V INV_PI

template<typename TPixelType>
GLM_FUNC_QUALIFIER TPixelType Texture2D<TPixelType>::getPixelFromSphereMap(glm::vec3 dir) const {
    //static const glm::vec2 invAtan(0.1591, 0.3183);
    dir = glm::normalize(dir);
    float phi = atan2(dir.x, -dir.z);
    float theta = asin(dir.y);
    float u = glm::fract(phi * SPHERE_MAP_FACTOR_U + 0.5f);
    float v = glm::fract(theta * SPHERE_MAP_FACTOR_V + 0.5f);
    //printf("dir<%f,%f,%f> -> (phi,theta)<%f,%f> -> uv<%f,%f>\n", dir.x, dir.y, dir.z, phi * 180.f / PI, theta * 180.f / PI, u, v);
    return getPixelByUVBilinear(u, v);
}
