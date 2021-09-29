#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "scenestruct/material.h"
#include "scenestruct/geometry.h"
#include "utilities.h"

#define FLIP_NORMAL_IF_INSIDE 0

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

__host__ __device__ inline glm::vec3 getPointOnRayPenetrate(Ray r, float t) {
    return r.origin + (t + .0001f) * glm::normalize(r.direction);
}

__host__ __device__ inline void updateOriginWithBias(Ray& r) {
    r.origin += r.direction 
        //* EPSILON;
        * .0001f;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ inline float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1.f : -1.f;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    //if (tmax >= tmin && tmax > 0) {
    if (tmax >= tmin && tmax > FLT_EPSILON) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
#if FLIP_NORMAL_IF_INSIDE
            tmin_n = tmax_n;
#else // FLIP_NORMAL_IF_INSIDE
            tmin_n = -tmax_n;
#endif // FLIP_NORMAL_IF_INSIDE
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ inline float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    //float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - radius * radius);
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
#if FLIP_NORMAL_IF_INSIDE
    if (!outside) {
        normal = -normal;
    }
#endif // FLIP_NORMAL_IF_INSIDE

    return glm::length(r.origin - intersectionPoint);
}

#define TRIANGLE_INTERSECTION_IN_WORLD_SPACE 1

__host__ __device__ inline float trimeshIntersectionTest(Geom trimeshgeom, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId) {
    if (!trimeshgeom.trimeshRes.isReadable()) {
        return -1.f;
    }

#if TRIANGLE_INTERSECTION_IN_WORLD_SPACE
    ///////////////////////////////

    float t = trimeshgeom.trimeshRes.worldIntersectionTest(trimeshgeom.transform, r, intersectionPoint, intersectionBarycentric, normal, triangleId);

    return t;

    ///////////////////////////////
#else // TRIANGLE_INTERSECTION_IN_WORLD_SPACE

    Ray q;
    q.origin    =                multiplyMV(trimeshgeom.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(trimeshgeom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = trimeshgeom.trimeshRes.localIntersectionTest(q, intersectionPoint, intersectionBarycentric, normal, triangleId);

    intersectionPoint = multiplyMV(trimeshgeom.transform, glm::vec4(intersectionPoint, 1.f));
    normal = glm::normalize(multiplyMV(trimeshgeom.invTranspose, glm::vec4(normal, 0.f)));
    return t;
#endif // TRIANGLE_INTERSECTION_IN_WORLD_SPACE
}

//__host__ __device__ inline float triangleIntersectionTest(Geom trigeom, Triangle triangle, Ray r,
//    glm::vec3 &intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3 &normal) {
//    Ray q;
//    q.origin    =                multiplyMV(trigeom.inverseTransform, glm::vec4(r.origin   , 1.0f));
//    q.direction = glm::normalize(multiplyMV(trigeom.inverseTransform, glm::vec4(r.direction, 0.0f)));
//
//    glm::vec3 e1 = triangle.pos1 - triangle.pos0, e2 = triangle.pos2 - triangle.pos0;
//    glm::vec3 s = q.origin - triangle.pos0;
//    glm::vec3 s1 = glm::cross(q.direction, e2), s2 = glm::cross(s, e1);
//
//    float s1_dot_e1 = glm::dot(s1, e1);
//    float s2_dot_e2 = glm::dot(s2, e2);
//    float s1_dot_s = glm::dot(s1, s);
//    float s2_dot_dir = glm::dot(s2, q.direction);
//
//    if (fabs(s2_dot_dir) < EPSILON) {
//        return -1.f;
//    }
//
//    if (!triangle.twoSided && s2_dot_dir < 0.f) {
//        return -1.f;
//    }
//
//    float tnear = s2_dot_e2 / s1_dot_e1;
//    float u = s1_dot_s / s1_dot_e1;
//    float v = s2_dot_dir / s1_dot_e1;
//    float w = 1.f - u - v;
//
//    if (tnear < EPSILON || u < 0.f || v < 0.f || w < 0.f) {
//        return -1.f;
//    }
//
//    intersectionBarycentric.x = u;
//    intersectionBarycentric.y = v;
//    intersectionBarycentric.z = w;
//
//    intersectionPoint = barycentricInterpolation(intersectionBarycentric, triangle.pos0, triangle.pos1, triangle.pos2);
//    normal = glm::normalize(barycentricInterpolation(intersectionBarycentric, triangle.nrm0, triangle.nrm1, triangle.nrm2));
//
//    intersectionPoint = multiplyMV(trigeom.transform, glm::vec4(intersectionPoint, 1.f));
//    normal = glm::normalize(multiplyMV(trigeom.invTranspose, glm::vec4(normal, 0.f)));
//    return tnear;
//}
