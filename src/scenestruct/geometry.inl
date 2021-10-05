#include "geometry.h"

/**
* Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
*/
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

GLM_FUNC_QUALIFIER BBox BBox::toWorld(const glm::mat4 transform) const {
    if (!isValid) {
        return *this;
    }

    glm::vec3 vertices[]{
        glm::vec3(multiplyMV(transform, glm::vec4(minP.x, minP.y, minP.z, 1.f))),
        glm::vec3(multiplyMV(transform, glm::vec4(minP.x, minP.y, maxP.z, 1.f))),
        glm::vec3(multiplyMV(transform, glm::vec4(minP.x, maxP.y, minP.z, 1.f))),
        glm::vec3(multiplyMV(transform, glm::vec4(minP.x, maxP.y, maxP.z, 1.f))),
        glm::vec3(multiplyMV(transform, glm::vec4(maxP.x, minP.y, minP.z, 1.f))),
        glm::vec3(multiplyMV(transform, glm::vec4(maxP.x, minP.y, maxP.z, 1.f))),
        glm::vec3(multiplyMV(transform, glm::vec4(maxP.x, maxP.y, minP.z, 1.f))),
        glm::vec3(multiplyMV(transform, glm::vec4(maxP.x, maxP.y, maxP.z, 1.f))),
    };
    glm::vec3 worldMinP = vertices[0], worldMaxP = vertices[0];
#pragma unroll
    for (size_t i = 1; i < 8; ++i) {
        worldMinP = glm::min(worldMinP, vertices[i]);
        worldMaxP = glm::max(worldMaxP, vertices[i]);
    }
    return {
        worldMinP,
        worldMaxP,
        1
    };
}

GLM_FUNC_QUALIFIER float BBox::intersectionTest(Ray q, bool& outside) const {
    if (!isValid) {
        return -1.f;
    }
    float tmin = -1e38f;
    float tmax = 1e38f;
    for (int component = 0; component < 3; ++component) {
        float qdxyz = q.direction[component];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (minP[component] - q.origin[component]) / qdxyz;
            float t2 = (maxP[component] - q.origin[component]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > 0 && ta > tmin) {
                tmin = ta;
            }
            if (tb < tmax) {
                tmax = tb;
            }
        }
    }

    //if (tmax >= tmin && tmax > 0) {
    if (tmax >= tmin && tmax > 0.f) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            outside = false;
        }
        return tmin;
    }
    return -1.f;
}

__host__ __device__ inline
glm::vec3 tangentSpaceToWorldSpace(const glm::vec3& dir, const glm::vec3& normal) {
    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return dir.z * normal
        + dir.x * perpendicularDirection1
        + dir.y * perpendicularDirection2;
}

__host__ __device__ inline
glm::vec3 calculateUniformRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine &rng, float* pdf = nullptr) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = u01(rng);
    float over = sqrt(1.f - up * up);
    float around = u01(rng) * TWO_PI;
    glm::vec3 dir(cos(around) * over, sin(around) * over, up);
    if (pdf) {
        *pdf = 1.f / TWO_PI;
    }
    return tangentSpaceToWorldSpace(dir, normal);
}

__host__ __device__ inline
glm::vec3 calculateCosWeightedRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine &rng, float* pdf = nullptr) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1.f - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;
    glm::vec3 dir(cos(around) * over, sin(around) * over, up);
    if (pdf) {
        *pdf = up / PI;
    }
    return tangentSpaceToWorldSpace(dir, normal);
}

__host__ __device__ inline
glm::vec3 calculateCosWeightedRandomDirectionInPhongSpecularRegion(
    glm::vec3 normal, thrust::default_random_engine &rng, float specex = 0.f, float* pdf = nullptr) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Reference: http://vclab.kaist.ac.kr/cs580/slide16-Path_tracing(2).pdf
    float up = specex <= 0.f ? 1.f : powf(u01(rng), 1.f / (specex + 1.f)); // cos(alpha)
    float over = sqrt(1.f - up * up); // sin(alpha)
    float around = u01(rng) * TWO_PI;
    glm::vec3 dir(cos(around) * over, sin(around) * over, up);
    if (pdf) {
        *pdf = specex <= 0.f ? 1.f : (specex + 1) * powf(up, specex) / TWO_PI;
    }
    return tangentSpaceToWorldSpace(dir, normal);
}

#if BUILD_BVH_FOR_TRIMESH
//template<>
//GLM_FUNC_QUALIFIER void BoundingVolumeHierarchy<Triangle>::buildBVH_CPU(Triangle* geoms, int geomNum, float expand) {
//    nodeNum = (geomNum << 1) - 1;
//    treeHeight = 0;
//    for (i32 i = 2; i - 2 < nodeNum; i <<= 1) { // 0(2)| 1(3), 2(4)| 3(5), 4(6), 5(7), 6(8)| ...
//        ++treeHeight;
//    }
//    BVHNode* nodesCPU;
//    printf("Initialize BVH with %d nodes, %d leaves, with height %d.\n", nodeNum, geomNum, treeHeight);
//    //BVHNode* 
//    cudaMemcpy
//    //cudaMemcpy(nodesArray, nodesCPU, sizeof(BVHNode) * nodeNum, cudaMemcpyHostToDevice);
//}

template<>
GLM_FUNC_QUALIFIER float BoundingVolumeHierarchy<Triangle>::worldIntersectionTest(
        const glm::mat4& transform, const glm::mat4& invTransform, const glm::mat4& invTranspose, 
        Ray r, Triangle* geoms, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, i32& geomId) const {
    constexpr size_t STACK_SIZE = 32;
    i32 stackForQuery[STACK_SIZE];
    i32 stackTopIdx = 0; // TOO MUCH ERROR FOR LOCAL INTERSECTION

    bool outside;

    i32 curIdx;
    glm::vec3 tmp_point, tmp_bary, tmp_nrm;

    float tmin = FLT_MAX;

    Ray q;
    q.origin    =                multiplyMV(invTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(invTransform, glm::vec4(r.direction, 0.0f)));

    stackForQuery[stackTopIdx++] = 0;
    while (stackTopIdx > 0) {
        curIdx = stackForQuery[stackTopIdx - 1];
        --stackTopIdx;
        BVHNode curNode = nodesArray[curIdx];
        if (curNode.box.intersectionTest(q, outside) > 0.f) {
        //if (curNode.box.toWorld(transform).intersectionTest(r, outside) > 0.f) {
            if (curNode.geomIdx < 0) {
                i32 rightIdx = rightChildIdx(curIdx);
                i32 leftIdx = leftChildIdx(curIdx);
                if (rightIdx > 0 && rightIdx < nodeNum) {
                    stackForQuery[stackTopIdx++] = rightIdx;
                }
                if (leftIdx > 0 && leftIdx < nodeNum) {
                    stackForQuery[stackTopIdx++] = leftIdx;
                }
            }
            else {
                Triangle geom = geoms[curNode.geomIdx].toWorld(transform, invTranspose);

                float tmp_t = geom.triangleLocalIntersectionTest(r, tmp_point, tmp_bary, tmp_nrm);
                if (tmp_t > 0.f && tmp_t < tmin) {
                    tmin = tmp_t;
                    intersectionPoint = tmp_point;
                    intersectionBarycentric = tmp_bary;
                    normal = tmp_nrm;
                    geomId = curNode.geomIdx;
                }
            }
        }
    }
    if (geomId < 0) {
        return -1.f;
    }
    return tmin;
}
#endif // BUILD_BVH_FOR_TRIMESH

template<>
GLM_FUNC_QUALIFIER BBox BBox::getLocalBoundingBox(const Triangle& geom, float expand) {
    return BBox{
        glm::min(geom.pos0, glm::min(geom.pos1, geom.pos2)) - expand,
        glm::max(geom.pos0, glm::max(geom.pos1, geom.pos2)) + expand,
        1
    };
}


template<typename TVec>
GLM_FUNC_QUALIFIER TVec barycentricInterpolation(const glm::vec3& bary, const TVec& v0, const TVec& v1, const TVec& v2) {
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

GLM_FUNC_QUALIFIER glm::vec4 getBarycentric(const glm::vec2& p, const glm::vec2& v0, const glm::vec2& v1, const glm::vec2& v2) {
    glm::vec2 e0 = p - v0, e1 = p - v1, e2 = p - v2;
    float alpha = e0.x * e1.y - e0.y * e1.x, beta = e1.x * e2.y - e1.y * e2.x, gamma = e2.x * e0.y - e2.y * e0.x;
    float sum = fabs(alpha + beta + gamma);
    if (sum < FLT_EPSILON) {
        return glm::vec4(1.f / 3.f, 1.f / 3.f, 1.f / 3.f, 0.f);
    }
    alpha /= sum;
    beta /= sum;
    gamma /= sum;
    return glm::vec4(alpha, beta, gamma, 1.f);
}

GLM_FUNC_QUALIFIER glm::vec4 getBarycentric(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
    glm::vec3 e0 = p - v0, e1 = p - v1, e2 = p - v2;
    glm::vec3 posAxis = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    glm::vec3 c01 = glm::cross(e0, e1), c12 = glm::cross(e1, e2), c20 = glm::cross(e2, e0);
    float alpha = glm::dot(c12, posAxis), beta = glm::dot(c20, posAxis), gamma = glm::dot(c01, posAxis);
    float sum = fabs(alpha + beta + gamma);
    if (sum < FLT_EPSILON) {
        return glm::vec4(1.f / 3.f, 1.f / 3.f, 1.f / 3.f, 0.f);
    }
    alpha /= sum;
    beta /= sum;
    gamma /= sum;
    return glm::vec4(alpha, beta, gamma, 1.f);
}

GLM_FUNC_QUALIFIER float TriMesh::worldIntersectionTest(
        const glm::mat4& transform, const glm::mat4& invTransform, const glm::mat4& invTranspose,
        Ray r, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId) {
#if BUILD_BVH_FOR_TRIMESH
    return localBVH.worldIntersectionTest(transform, invTransform, invTranspose, r, triangles, intersectionPoint, intersectionBarycentric, normal, triangleId);
#else // BUILD_BVH_FOR_TRIMESH
    float tnear = FLT_MAX;
    float tmp_t = -1.f;
    int tmp_triangleId = -1;
    glm::vec3 tmp_p, tmp_b, tmp_n;
    for (int i = 0; i < triangleNum; ++i) {
        Triangle tri = triangles[i].toWorld(transform, invTranspose);

        tmp_t = tri.triangleLocalIntersectionTest(r, tmp_p, tmp_b, tmp_n);
        if (tmp_t > 0.f && tmp_t < tnear) {
            tmp_triangleId = i;
            intersectionPoint = tmp_p;
            intersectionBarycentric = tmp_b;
            normal = tmp_n;
            tnear = tmp_t;
        }
}
    if (tmp_triangleId == -1) {
        return -1.f;
    }
    triangleId = tmp_triangleId;
    return tnear;
#endif // BUILD_BVH_FOR_TRIMESH
}

//GLM_FUNC_QUALIFIER float TriMesh::localIntersectionTest(Ray q, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId) {
//#if BUILD_BVH_FOR_TRIMESH
//
//#else // BUILD_BVH_FOR_TRIMESH
//    float tnear = FLT_MAX;
//    float tmp_t = -1.f;
//    int tmp_triangleId = -1;
//    for (int i = 0; i < triangleNum; ++i) {
//        tmp_t = triangles[i].triangleLocalIntersectionTest(q, intersectionPoint, intersectionBarycentric, normal);
//        if (tmp_t > 0.f && tmp_t < tnear) {
//            tmp_triangleId = i;
//            tnear = tmp_t;
//        }
//    }
//    if (tmp_triangleId == -1) {
//        return -1.f;
//    }
//    triangleId = tmp_triangleId;
//    return tnear;
//#endif // BUILD_BVH_FOR_TRIMESH
//}

#define TRIANGLE_INTERSECTION_EPSILON 0.f //0.005f // 0.001f

GLM_FUNC_QUALIFIER Triangle Triangle::toWorld(const glm::mat4& transform, const glm::mat4& invTranspose) const
{
    Triangle tri(*this);
    tri.pos0 = multiplyMV(transform, glm::vec4(pos0, 1.f));
    tri.pos1 = multiplyMV(transform, glm::vec4(pos1, 1.f));
    tri.pos2 = multiplyMV(transform, glm::vec4(pos2, 1.f));

    //tri.nrm0 = glm::normalize(multiplyMV(transform, glm::vec4(nrm0, 0.f)));
    //tri.nrm1 = glm::normalize(multiplyMV(transform, glm::vec4(nrm1, 0.f)));
    //tri.nrm2 = glm::normalize(multiplyMV(transform, glm::vec4(nrm2, 0.f)));
    tri.nrm0 = glm::normalize(multiplyMV(invTranspose, glm::vec4(nrm0, 0.f)));
    tri.nrm1 = glm::normalize(multiplyMV(invTranspose, glm::vec4(nrm1, 0.f)));
    tri.nrm2 = glm::normalize(multiplyMV(invTranspose, glm::vec4(nrm2, 0.f)));

    return tri;
}

GLM_FUNC_QUALIFIER float Triangle::triangleLocalIntersectionTest(Ray q, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal) {
    glm::vec3 e1 = pos1 - pos0, e2 = pos2 - pos0;
    glm::vec3 s = q.origin - pos0;
    glm::vec3 s1 = glm::cross(q.direction, e2), s2 = glm::cross(s, e1);

    float s1_dot_e1 = glm::dot(s1, e1);
    float s2_dot_e2 = glm::dot(s2, e2);
    float s1_dot_s = glm::dot(s1, s);
    float s2_dot_dir = glm::dot(s2, q.direction);

    if (fabs(s2_dot_dir) < EPSILON) {
        return -1.f;
    }

    if (!twoSided && s2_dot_dir < 0.f) {
        return -1.f;
    }

    float tnear = s2_dot_e2 / s1_dot_e1;
    float u = s1_dot_s / s1_dot_e1;
    float v = s2_dot_dir / s1_dot_e1;
    float w = 1.f - u - v;

    if (tnear < EPSILON || u < -TRIANGLE_INTERSECTION_EPSILON || v < -TRIANGLE_INTERSECTION_EPSILON || w < -TRIANGLE_INTERSECTION_EPSILON) {
        return -1.f;
    }
    //if (tnear < EPSILON || u < 0.f || v < 0.f || w < 0.f) {
    //    return -1.f;
    //}

    intersectionBarycentric.x = w;
    intersectionBarycentric.y = u;
    intersectionBarycentric.z = v;

    intersectionPoint = barycentricInterpolation(intersectionBarycentric, pos0, pos1, pos2);
    normal = glm::normalize(barycentricInterpolation(intersectionBarycentric, nrm0, nrm1, nrm2));
    if (twoSided && s2_dot_dir < 0.f) {
        normal = -normal;
    }
    return tnear;
}

