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

// modified from the thrust version to return true for hitting
// the back of a face
__host__ __device__ bool intersectRayTriangle
	(
		glm::vec3 const & orig, glm::vec3 const & dir,
		glm::vec3 const & v0, glm::vec3 const & v1, glm::vec3 const & v2,
		glm::vec3 & baryPosition
	)
	{
		glm::vec3 e1 = v1 - v0;
		glm::vec3 e2 = v2 - v0;

		glm::vec3 p = glm::cross(dir, e2);

		float a = glm::dot(e1, p);

		float Epsilon = FLT_EPSILON;

        // this part of glm's triangle intersection function rules out
        // hitting the back of a face. We want to count that as a hit for
        // refraction and meshes with holes (like the teapot).
        // using the floating point abs of a doesn't rule out negative
        // results of the above dot product, and therefore rule in
        // the back face of polys
		if(fabs(a) < Epsilon)
			return false;

		float f = float(1.0f) / a;

		glm::vec3 s = orig - v0;
		baryPosition.x = f * glm::dot(s, p);
		if(baryPosition.x < float(0.0f))
			return false;
		if(baryPosition.x > float(1.0f))
			return false;

		glm::vec3 q = glm::cross(s, e1);
		baryPosition.y = f * glm::dot(dir, q);
		if(baryPosition.y < float(0.0f))
			return false;
		if(baryPosition.y + baryPosition.x > float(1.0f))
			return false;

		baryPosition.z = f * glm::dot(e2, q);

		return baryPosition.z >= float(0.0f);
	}


__host__ __device__ float meshIntersectionTest(Geom geom, 
											   Ray r,
											   glm::vec3 &intersectionPoint, 
											   glm::vec3 &normal, 
											   bool &outside,
                                               Tri * tris,
                                               int numTris,
                                               Tri * bboxTris,
                                               bool useBBox) {
    Ray q;
    q.origin =    multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));


    // --- check for intersections with the mesh's bounding box
	glm::vec3 baryPos;
    bool hit = false;
    Tri tri;
    if (useBBox) {
        for (int i = 0; i < 12; i++) {
            tri = bboxTris[i];

            // don't use glm's intersect RayTriangle,
            // it doesn't detect back face intersections
            hit = intersectRayTriangle(q.origin,
                q.direction,
                tri.v1,
                tri.v2,
                tri.v3,
                baryPos);

            if (hit) {
                break;
            }
        }
        if (!hit) {
            return -1;
        }
    }


    // --- iterate over the triangles in our mesh looking for intersections
    float tmin = 1e38f;
    glm::vec3 tmin_n;
	float t; 
    for (int i = 0; i < numTris; i++) {
        tri = tris[i];
        glm::vec3 v1 = tri.v1;
        glm::vec3 v2 = tri.v2;
        glm::vec3 v3 = tri.v3;
        hit = false;

        hit = intersectRayTriangle(q.origin,
									    q.direction,
									    v1,
									    v2,
									    v3,
									    baryPos);

        //volatile float3 bp = make_float3(baryPos.x, baryPos.y, baryPos.z);
        if (hit) {
            // glm::intersect doesn't actually give us all barycentric 
            // values, only u & v. We can calculate w using these though
            float w = (1.0f - baryPos.x - baryPos.y);
            // the third value from glm::intersect is actually the t we're 
            // looking for though, so that's nice
            t = baryPos.z;

            if (t < tmin){
                tmin = t;
                tmin_n = baryPos.x * tri.n1 + baryPos.y * tri.n2 + w * tri.n3;
            }
        }
    }

    if (tmin < 1e38f) {
        intersectionPoint = multiplyMV(geom.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        // transform the normal we found to worldspace, but if it's the back of the poly, just flip the normal
        normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(tmin_n, 0.0f)) * -1.0f * dot(q.direction, tmin_n));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}
