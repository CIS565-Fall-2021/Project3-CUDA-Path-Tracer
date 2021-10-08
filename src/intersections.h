#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "interactions.h"

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

/**
 * Test intersection between a ray and an aabb.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */

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
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n(0.f);
            n[xyz] = t2 < t1 ? +1.f : -1.f;
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
        t = glm::min(t1, t2);
        outside = true;
    }
    else
    {
        t = glm::max(t1, t2);
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

// triangle degined by vertices v0, v1 and  v2
// returns (t, u, v) where p = v0 + u(v1-v0) + v(v2-v0)
__host__ __device__ glm::vec3 triIntersect(Ray const &r, Triangle const &tri)
{
    using namespace glm;
    vec3 ro = r.origin;
    vec3 rd = normalize(r.direction);
    vec3 vertA = tri.pos[0];
    vec3 vertB = tri.pos[1];
    vec3 vertC = tri.pos[2];
    vec3 aToB = vertB - vertA;
    vec3 aToC = vertC - vertA;
    vec3 aToRo = ro - vertA;

    // q, intersect point in triangle, = v0 + alpha(v1-v0) + beta(v2-v0)
    // ray is p + td where p is point and d is direction
    // Mx = s -> M = {-d, v1-v0, v2-v0},
    //           x = {p-a},
    //           s = {t, alpha, beta}Transpose

    // vec3 triNorm = cross(aToB, aToC); // Triangle normal from points
    // norm = triNorm;
    vec3 triNorm = tri.planarNorm;
    vec3 q = cross(aToRo, rd); // used for scalar triple product
    float negDet = dot(rd, triNorm);
    float negReciporicalDet = 1.f / negDet;
    float t = negReciporicalDet * -1.f * dot(triNorm, aToRo);
    float u = negReciporicalDet * -1.f * dot(q, aToC);
    float v = negReciporicalDet * dot(q, aToB);
    if (u < 0.f || u > 1.f || v < 0.f || (u + v) > 1.f)
    {
        t = -1.f;
    }
    return vec3(t, u, v);
}

__host__ __device__ float triangleIntersectionTest(Geom const &tri, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2 &uv, struct Triangle *tris, int index)
{
    Ray q; // in triangle space
    q.origin = multiplyMV(tri.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(tri.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // glm::vec3 norm;
    struct Triangle t = tris[index];
    // glm::vec3 tuv = triIntersect(q, t, norm);
    glm::vec3 tuv = triIntersect(q, t);
    if (tuv.x <= 0.f)
    {
        return -1.f;
    }
    outside = glm::dot(t.planarNorm, q.direction) < 0.f;
    // back from triangle space
    intersectionPoint = multiplyMV(tri.transform, glm::vec4(getPointOnRay(q, tuv.x), 1.f));
    uv = (1.f - tuv.y - tuv.z) * t.uv[0] +
         tuv.y * t.uv[1] +
         tuv.z * t.uv[2];
    glm::vec3 normalTri(
        (1.f - tuv.y - tuv.z) * t.norm[0] +
        tuv.y * t.norm[1] +
        tuv.z * t.norm[2]);
    // glm::Vec3 normalTri()
    normalTri *= (outside ? 1.f : -1.f);
    normal = glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(normalTri, 0.f)));
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ __forceinline__ void swap(float &a, float &b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

__host__ __device__ bool aabbIntersectionTest(Geom const g, Ray r)
{
    /*
    x0 + tx = xn
    aabbmin.x - qo.x then / qd.x -> new tmin
    aabbmax.x - qo.x then / qd.x -> new tmax
    */
    Ray q;
    q.origin = multiplyMV(g.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(g.inverseTransform, glm::vec4(r.direction, 0.0f)));
    float tmpMin, tmin = 0.f;
    float tmpMax, tmax = 1e38f;
    glm::vec3 aabbmin = g.min;
    glm::vec3 aabbmax = g.max;
    // Check collisions if ray starts inside
    if (q.origin.x <= aabbmax.x &&
        q.origin.y <= aabbmax.y &&
        q.origin.z <= aabbmax.z &&
        q.origin.x >= aabbmin.x &&
        q.origin.y >= aabbmin.y &&
        q.origin.z >= aabbmin.z)
    {
        return true;
    }

    for (int i = 0; i < 3; i++)
    {
        if (q.direction[i] == 0)
        {
            if (q.origin[i] > aabbmax[i] || q.origin[i] < aabbmin[i])
            {
                return false;
            }
            else
            {
                continue;
            }
        }
        else
        {
            tmpMin = (aabbmin[i] - q.origin[i]) / q.direction[i];
            tmpMax = (aabbmax[i] - q.origin[i]) / q.direction[i];
            if (tmpMin > tmpMax)
            {
                swap(tmpMin, tmpMax);
            }
        }
        tmin = glm::max(tmin, tmpMin);
        tmax = glm::min(tmax, tmpMax);
    }
    return tmin <= tmax;
}

__host__ __device__ float meshIntersectionTest(Geom const &mesh, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, glm::vec2 &uv, struct Triangle *tri, glm::mat3 &tbn)
{
#ifdef BV_CULL
    if (!aabbIntersectionTest(mesh, r))
    {
        return -1.f;
    }
    else
#endif
    {
        float t = 1e38f;
        glm::vec3 tmpIsec, tmpNorm;
        glm::vec2 tmpUv;
        bool tmpOut;
        float tmpT;
        for (int i = 0; i < mesh.numTris; i++)
        {
            int idx = i + mesh.triIdx;
            tmpT = triangleIntersectionTest(mesh, r, tmpIsec, tmpNorm, tmpOut, tmpUv, tri, idx);
            if (tmpT < t && tmpT > 0.f)
            {
                intersectionPoint = tmpIsec;
                normal = tmpNorm;
                outside = tmpOut;
                uv = tmpUv;
                t = tmpT;
                tbn = glm::mat3(tri[idx].tangent, tri[idx].bitangent, tri[idx].planarNorm);
            }
        }
        return t == 1e38f ? -1.f : t;
    }
}
