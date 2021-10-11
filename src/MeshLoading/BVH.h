#pragma once

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "../sceneStructs.h"


class BVH
{
  
public:
    static const int kNumPlaneSetNormals = 7;

    static const glm::vec3 planeSetNormals[kNumPlaneSetNormals];
    struct Extents
    {
        float d[kNumPlaneSetNormals][2];
        Extents()
        {
            for (uint8_t i = 0; i < kNumPlaneSetNormals; ++i)
                d[i][0] = INT_MAX, d[i][1] = INT_MIN;
        }
    };

    Extents extents;
    int triangleCount;
    glm::vec4 *Triangle_Point_Normals;
    BVH(int a_triangleCount, glm::vec4* a_Triangle_Point_Normals)
    {
        triangleCount = a_triangleCount;
        Triangle_Point_Normals = a_Triangle_Point_Normals;
    }
    void GetBounds(float *getBounds, Geom &geom);
    ~BVH();
};

