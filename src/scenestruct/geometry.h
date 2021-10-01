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
    glm::vec3 minP, maxP;
    ui8 isValid = 0;

    template<typename TGeom>
    GLM_FUNC_QUALIFIER static BBox getLocalBoundingBox(const TGeom& geom, float expand = 0.f) {
        return BBox();
    }

    GLM_FUNC_QUALIFIER BBox operator +(const BBox& b) {
        BBox b1(*this);
        b1 += b;
        return b1;
    }

    GLM_FUNC_QUALIFIER BBox& operator +=(const BBox& b) {
        if (!isValid) {
            minP = b.minP;
            maxP = b.maxP;
            isValid = b.isValid;
        }
        else if (b.isValid) {
            minP = glm::min(minP, b.minP);
            maxP = glm::max(maxP, b.maxP);
        }
        return *this;
    }

    GLM_FUNC_QUALIFIER i32 getMaxDistAxis() const {
        i32 result = 0;
        float dist = maxP[0] - minP[0];
        float dist1 = maxP[1] - minP[1];
        if (dist1 > dist) {
            result = 1;
            dist = dist1;
        }
        dist1 = maxP[2] - minP[2];
        if (dist1 > dist) {
            result = 2;
            dist = dist1;
        }
        return result;
    }

    GLM_FUNC_QUALIFIER glm::vec3 getCenter() const {
        return (minP + maxP) * 0.5f;
    }

    GLM_FUNC_QUALIFIER BBox toWorld(const glm::mat4 transform) const;

    GLM_FUNC_QUALIFIER float intersectionTest(Ray q, bool& outside) const;
};

struct Triangle {
    GLM_FUNC_QUALIFIER Triangle() {}
    GLM_FUNC_QUALIFIER Triangle(const Triangle& tri) {
        memcpy(this, &tri, sizeof(tri));
    }
    GLM_FUNC_QUALIFIER Triangle& operator=(const Triangle& tri) {
        memcpy(this, &tri, sizeof(tri));
        return *this;
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

    GLM_FUNC_QUALIFIER Triangle toWorld(const glm::mat4& transform, const glm::mat4& invTranspose) const;

    GLM_FUNC_QUALIFIER float triangleLocalIntersectionTest(Ray q, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal);
};

#if BUILD_BVH_FOR_TRIMESH
template<typename TGeom>
struct BoundingVolumeHierarchy {
    struct BVHNode {
        BBox box;
        i32 leftSubtreeIdx = -1, rightSubtreeIdx = -1;
        i32 geomIdx = -1;
    };

    GLM_FUNC_QUALIFIER i32 leftChildIdx(i32 root) const { 
        return nodesArray[root].leftSubtreeIdx;
        //return (root << 1) + 1; 
    }
    GLM_FUNC_QUALIFIER i32 rightChildIdx(i32 root) const { 
        return nodesArray[root].rightSubtreeIdx;
        //return (root << 1) + 2; 
    }
    //GLM_FUNC_QUALIFIER i32 nextNodeIdx(i32 curr) const {
    //    i32 level = 0;

    //    for (i32 i = 2; i - 2 < curr; i <<= 1) { // 0(2)| 1(3), 2(4)| 3(5), 4(6), 5(7), 6(8)| ...
    //        ++level;
    //    }

    //    i32 base = 1 << level;

    //    do {
    //        if (curr & 1 == 0) {
    //            // Left node

    //        }
    //    } while(curr < nodeNum && nodesArray[curr].geomIdx == -1);
    //    return curr;
    //}

    __host__ void buildBVH_CPU(TGeom* geoms, i32 geomNum, float expand = EPSILON) {}

    GLM_FUNC_QUALIFIER float worldIntersectionTest(
        const glm::mat4& transform, const glm::mat4& invTransform, const glm::mat4& invTranspose, 
        Ray r, TGeom* geoms, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, i32& geomId) const { return -1; }

    i32 nodeNum = 0, treeHeight = 0;
    BVHNode* nodesArray = nullptr;
};
#endif // BUILD_BVH_FOR_TRIMESH

struct TriMesh {
    Triangle* triangles;
    i32 triangleNum = 0;

#if BUILD_BVH_FOR_TRIMESH
    BoundingVolumeHierarchy<Triangle> localBVH;
#endif // BUILD_BVH_FOR_TRIMESH

    GLM_FUNC_QUALIFIER bool isReadable() const {
        return triangles != nullptr;
    }
    GLM_FUNC_QUALIFIER float worldIntersectionTest(
        const glm::mat4& transform, const glm::mat4& invTransform, const glm::mat4& invTranspose, 
        Ray r, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId);
    //GLM_FUNC_QUALIFIER float localIntersectionTest(Ray r, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId);
};

struct Geom {
    enum GeomType type = GeomType::NONE;
    int geometryid = -1;
    int materialid = -1;
    int stencilid = 0;
    TriMesh trimeshRes;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

#include "geometry.inl"
