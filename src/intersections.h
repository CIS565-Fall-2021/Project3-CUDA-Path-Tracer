#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

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
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
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
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
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
            n[xyz] = t2 < t1 ? +1 : -1;
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

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
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
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
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
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

float TriangleArea(glm::vec4 a_p1, glm::vec4 a_p2, glm::vec4 a_p3)
{
    float A = 0.5f * glm::length(glm::cross(glm::vec3(a_p2[0] - a_p1[0], a_p2[1] - a_p1[1], 0),
        glm::vec3(a_p3[0] - a_p1[0], a_p3[1] - a_p1[1], 0)));
    return A;
}


glm::vec4 GetBarycentricWeightedNormal(Vertex a_p1, Vertex a_p2, Vertex a_p3, glm::vec4 a_p)
{

    float A = TriangleArea(a_p1.m_pos, a_p2.m_pos, a_p3.m_pos);
    float A1 = TriangleArea(a_p2.m_pos, a_p3.m_pos, a_p);
    float A2 = TriangleArea(a_p1.m_pos, a_p3.m_pos, a_p);
    float A3 = TriangleArea(a_p1.m_pos, a_p2.m_pos, a_p);
    glm::vec4 a_surfaceNormal = a_p[2] * ((a_p1.m_normal * A1) / (A * a_p1.m_pos[2]) + (a_p2.m_normal * A2) / (A * a_p2.m_pos[2]) + (a_p3.m_normal * A3) / (A * a_p3.m_pos[2]));
    return a_surfaceNormal;
}


__host__ __device__ float MeshIntersectionTest(Geom objGeom, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    
    bool intersection = false;
    glm::vec3 interPoint;
    glm::vec3 internormal;
    int count = objGeom.triangleCount;

    for (int i = 0; i < count; i++)
    {
        //glm::vec4 p1 = glm::vec3(multiplyMV(objGeom.transform, objGeom.meshTriangles[i].points[0]));
        //glm::vec4 p2 = glm::vec3(multiplyMV(objGeom.transform, objGeom.meshTriangles[i].points[1]));
        //glm::vec4 p3 = glm::vec3(multiplyMV(objGeom.transform, objGeom.meshTriangles[i].points[2]));

        //glm::mat4 modelMat = objGeom.transform;
        glm::vec4 p1 = objGeom.transform * objGeom.meshTriangles[i].points[0];
        glm::vec4 p2 = objGeom.transform * objGeom.meshTriangles[i].points[1];
        glm::vec4 p3  = objGeom.transform * objGeom.meshTriangles[i].points[2];


       //  TriangleCustom abc = objGeom.meshTriangles[i];
        //glm::vec4 p1 = objGeom.transform * glm::vec4(1, 1, 1, 1);
        //glm::vec4 p2 = objGeom.transform * glm::vec4(1, 2, 1, 1);
        //glm::vec4 p3  = objGeom.transform * glm::vec4(2, 1, 1, 1);

        intersection = glm::intersectRayTriangle(r.origin, r.direction, glm::vec3(p1), glm::vec3(p2), glm::vec3(p3), interPoint);



        if (intersection)
        {

            Vertex v1(p1, glm::vec3(0, 0, 0), objGeom.meshTriangles[i].normals[0], glm::vec2(0,0));
            Vertex v2(p2, glm::vec3(0, 0, 0), objGeom.meshTriangles[i].normals[1], glm::vec2(0,0));
            Vertex v3(p3, glm::vec3(0, 0, 0), objGeom.meshTriangles[i].normals[2], glm::vec2(0,0));
            internormal = glm::vec3(GetBarycentricWeightedNormal(v1, v2, v3, glm::vec4(interPoint, 1.0f)));
            break;
        }
    }
    if (intersection)
    {
        intersectionPoint = interPoint;
        normal = internormal;
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

