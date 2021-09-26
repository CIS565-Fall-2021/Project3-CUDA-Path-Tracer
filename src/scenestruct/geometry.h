#pragma once

#include "glm/glm.hpp"
#include "../utilities.h"

#define BUILD_BVH_FOR_TRIMESH ENABLE_BVH


namespace RayRemainingBounce {
    constexpr int FIND_EMIT_SOURCE = -1;
    constexpr int OUT_OF_SCENE = -2;
}

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

enum class GeomType : ui8 {
    NONE,
    SPHERE,
    CUBE,
    TRI_MESH,
};

struct BBox {
    glm::vec3 min, max;
    int isValid = 0;

    template<typename TGeom>
    GLM_FUNC_QUALIFIER static BBox getLocalBoundingBox(const TGeom& geom) {
        return BBox();
    }
};

struct Triangle {
    GLM_FUNC_QUALIFIER Triangle() {}
    GLM_FUNC_QUALIFIER Triangle(const Triangle& tri) {
        memcpy(this, &tri, sizeof(tri));
    }

    union {
        struct { glm::vec3 pos0, pos1, pos2; };
        glm::vec3 position[3];
    };
    union {
        struct { glm::vec3 nrm0, nrm1, nrm2; };
        glm::vec3 normal[3];
    };
    union {
        struct { glm::vec2 uv00, uv01, uv02; };
        glm::vec2 uv0[3];
    };
    int triangleid = 0;
    ui8 twoSided = 0;


    GLM_FUNC_QUALIFIER float triangleLocalIntersectionTest(Ray q, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal);
};

#if BUILD_BVH_FOR_TRIMESH
template<typename TGeom>
struct BoundingVolumeHierarchy {
    GLM_FUNC_QUALIFIER void buildBVH(TGeom* dev_geom) {}

    int nodeSize = 0;
};
#endif // BUILD_BVH_FOR_TRIMESH

struct TriMesh {
    Triangle* triangles;
    int triangleNum = 0;

#if BUILD_BVH_FOR_TRIMESH
    BoundingVolumeHierarchy<TriMesh> localBVH;
#endif // BUILD_BVH_FOR_TRIMESH

    GLM_FUNC_QUALIFIER bool isReadable() const {
        return triangles != nullptr;
    }
    GLM_FUNC_QUALIFIER float worldIntersectionTest(glm::mat4 transform, Ray r, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId);
    GLM_FUNC_QUALIFIER float localIntersectionTest(Ray r, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId);
};

struct Geom {
    enum GeomType type = GeomType::NONE;
    int materialid;
    TriMesh trimeshRes;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

#include "geometry.inl"
