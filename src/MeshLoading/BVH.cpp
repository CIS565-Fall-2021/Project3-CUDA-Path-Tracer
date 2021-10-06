#include "BVH.h"


const glm::vec3 BVH::planeSetNormals[BVH::kNumPlaneSetNormals] = {
    glm::vec3(1, 0, 0),
    glm::vec3(0, 1, 0),
    glm::vec3(0, 0, 1),
    glm::vec3(sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f),
    glm::vec3(-sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f),
    glm::vec3(-sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f),
    glm::vec3(sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f) };

BVH::~BVH(){}

void computeBounds(int triangleCount, glm::vec4* Triangle_Point_Normals, const glm::vec3& planeNormal, float& dnear, float& dfar, Geom &geom)
{

    float d;
    for (uint32_t i = 0; i < triangleCount; ++i) {
        for (int j = 0; j < 3; j++)
        {
            glm::vec3 transformedPoint = glm::vec3(geom.transform *  Triangle_Point_Normals[6 * i + 2 * j]);
            d = glm::dot(planeNormal, transformedPoint);
            if (d < dnear) dnear = d;
            if (d > dfar) dfar = d;
        }
    }
}

void BVH::GetBounds(float* getBounds, Geom &geom)
{
    for (uint8_t j = 0; j < this->kNumPlaneSetNormals; ++j) {
        computeBounds(this->triangleCount, this->Triangle_Point_Normals, this->planeSetNormals[j], this->extents.d[j][0], this->extents.d[j][1], geom);
    }

    for (int i = 0; i < this->kNumPlaneSetNormals; i++)
    {
        getBounds[i] = this->extents.d[i][0];
        getBounds[i + 1] = this->extents.d[i][1];
    }
}