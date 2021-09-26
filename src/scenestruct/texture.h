#pragma once

#include "glm/glm.hpp"
#include "../utilities.h"

template<typename TPixelType>
struct Texture2D {
    GLM_FUNC_QUALIFIER operator bool() const;
    GLM_FUNC_QUALIFIER bool isReadable() const;
    GLM_FUNC_QUALIFIER TPixelType getPixel(int x, int y) const;
    GLM_FUNC_QUALIFIER TPixelType getPixelByUVBilinear(float u, float v) const;

    TPixelType* buffer = nullptr;
    glm::ivec2 size;
};

template<typename TPixelType>
GLM_FUNC_QUALIFIER Texture2D<TPixelType>::operator bool() const {
    return isReadable();
}

template<typename TPixelType>
GLM_FUNC_QUALIFIER bool Texture2D<TPixelType>::isReadable() const {
    return buffer != nullptr;
}

template<typename TPixelType>
GLM_FUNC_QUALIFIER TPixelType Texture2D<TPixelType>::getPixel(int x, int y) const {
    return buffer[y * size.x + x];
}

template<typename TPixelType>
GLM_FUNC_QUALIFIER TPixelType Texture2D<TPixelType>::getPixelByUVBilinear(float u, float v) const {
    int width = size.x, height = size.y;
    float u_img = u * width;
    float v_img = v * height;
    //float v_img = (1 - v) * height;

    float u_img_f = glm::max(0.f, floor(u_img));
    float u_img_c = glm::min(static_cast<float>(width - 1), ceil(u_img));
    float v_img_f = glm::max(0.f, floor(v_img));
    float v_img_c = glm::min(static_cast<float>(height - 1), ceil(v_img));

    int ufi = static_cast<int>(u_img_f + 0.5f);
    int uci = static_cast<int>(u_img_c + 0.5f);
    int vfi = static_cast<int>(v_img_f + 0.5f);
    int vci = static_cast<int>(v_img_c + 0.5f);

    auto color00 = getPixel(vfi, ufi);
    auto color01 = getPixel(vci, ufi);
    auto color10 = getPixel(vfi, uci);
    auto color11 = getPixel(vci, uci);

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
