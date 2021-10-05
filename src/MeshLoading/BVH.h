#pragma once

#pragma once
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "../sceneStructs.h"


class BVH
{
    static const int kNumPlaneSetNormals = 7;
    static const glm::vec3 planeSetNormals[kNumPlaneSetNormals];
    struct Extents
    {
        Extents()
        {
            for (uint8_t i = 0; i < kNumPlaneSetNormals; ++i)
                d[i][0] = INT_MAX, d[i][1] = INT_MIN;
        }
        float d[kNumPlaneSetNormals][2];
    };
    Extents extents;
public:
    int triangleCount;
    glm::vec4 *Triangle_Point_Normals;
    BVH(int a_triangleCount, glm::vec4* a_Triangle_Point_Normals)
    {
        triangleCount = a_triangleCount;
        Triangle_Point_Normals = a_Triangle_Point_Normals;
    }
    void GetBounds(float *getBounds);
    ~BVH();
};


const glm::vec3 BVH::planeSetNormals[BVH::kNumPlaneSetNormals] = {
    glm::vec3(1, 0, 0),
    glm::vec3(0, 1, 0),
    glm::vec3(0, 0, 1),
    glm::vec3(sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f),
    glm::vec3(-sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f),
    glm::vec3(-sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f),
    glm::vec3(sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f) };

void BVH::GetBounds(float* getBounds)
{
    for (uint8_t j = 0; j < this->kNumPlaneSetNormals; ++j) {
        computeBounds(this->triangleCount, this->Triangle_Point_Normals, this->planeSetNormals[j], this->extents.d[j][0], this->extents.d[j][1]);
    }

    for (int i = 0; i < this->kNumPlaneSetNormals; i++)
    {
        getBounds[i] = this->extents.d[i][0];
        getBounds[i+ 1] = this->extents.d[i][1];
    }
}


void computeBounds(int triangleCount, glm::vec4 *Triangle_Point_Normals, const glm::vec3 &planeNormal, float& dnear, float& dfar)
{

    float d;
    for (uint32_t i = 0; i < triangleCount; ++i) {
        for (int j = 0; j < 3; j++)
        {
            d = glm::dot(planeNormal, glm::vec3(Triangle_Point_Normals[6 * i + 2 * j]));
            if (d < dnear) dnear = d;
            if (d > dfar) dfar = d;
        }
    }
}
