#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define MESH_CULLING

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
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
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
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
                                              glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
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
                                                 glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float TriArea(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3)
{
    return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
}

__host__ __device__ glm::vec3 GetNormal(const Triangle &tri, const glm::vec3 P)
{
    float A = TriArea(tri.vertices[0], tri.vertices[1], tri.vertices[2]);
    float A0 = TriArea(tri.vertices[1], tri.vertices[2], P);
    float A1 = TriArea(tri.vertices[0], tri.vertices[2], P);
    float A2 = TriArea(tri.vertices[0], tri.vertices[1], P);
    return glm::normalize(tri.normals[0] * A0 / A + tri.normals[1] * A1 / A + tri.normals[2] * A2 / A);
}

__host__ __device__ float triangleIntersectionTest(Geom &mesh, Triangle &triangle, Ray rt, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside)
{
   

    glm::vec3 baryPos;

    bool hit = glm::intersectRayTriangle(rt.origin, rt.direction,
                                         triangle.vertices[0],
                                         triangle.vertices[1],
                                         triangle.vertices[2], baryPos);

    if (!hit)
    {
        return -1;
    }
    float t = baryPos.z;//glm::length(baryPos - rt.origin);
    intersectionPoint = getPointOnRay(rt, t);
    float baryZ = baryPos.z;
    baryPos.z = 1.0f - baryPos.x - baryPos.y;

     normal = glm::normalize(triangle.normals[0] * baryPos.z + triangle.normals[1] * baryPos.x + triangle.normals[2] * baryPos.y);
    // glm::vec3 newpos = triangle.vertices[0] * baryPos.z + triangle.vertices[1] * baryPos.x + triangle.vertices[2] * baryPos.y;
    // normal = triangle.normals[0];
    //normal = GetNormal(triangle, newpos);
    //normal = glm::normalize(triangle.n);

    // normal = glm::normalize(glm::cross(triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[1]));
  //   outside = glm::dot(normal, -rt.direction) > 0.f;
   //  normal *= outside ? 1.f : -1.f;
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normal, 0.f)));

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.0f));

    return t;
}

__host__ __device__ bool boundingBoxCheck(Geom mesh, Ray q)
{
    // Ray q;
    // q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    // q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (mesh.minBounding[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (mesh.maxBounding[xyz] - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    return (tmax >= tmin && tmax > 0);
}

__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, Triangle *triangles)
{
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
#ifdef MESH_CULLING
    bool hit = boundingBoxCheck(mesh, rt);
    if (!hit)
    {
        return -1;
    }
#endif

    float min_t = FLT_MAX;
    glm::vec3 tempIsectPoint;
    glm::vec3 tempNormal;
    for (int i = mesh.triStartIdx; i < mesh.triEndIdx; i++)
    {
        float t = triangleIntersectionTest(mesh, triangles[i], rt, tempIsectPoint, tempNormal, outside);
        if (t > 0.0f && t < min_t)
        {
            min_t = t;
            intersectionPoint = tempIsectPoint;
            normal = tempNormal;
        }
    }

    if (glm::abs(min_t - FLT_MAX) < 0.001)
    {
        return -1;
    }
    return min_t;
}