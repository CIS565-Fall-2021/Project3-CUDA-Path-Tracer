#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "cu.h"

#define BOUNDING_BOX 0

/* Handy-dandy hash function that provides seeds for random number generation. */
__host__ __device__ inline unsigned int hash(unsigned int a)
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


/* for triangle meshes */
__host__ __device__ float triangle_intersection_test(const Triangle tri, const Ray r, glm::vec3 *intersect, glm::vec3 *normal)
{
	glm::vec3 b_coord;
	if (!glm::intersectRayTriangle(r.origin, r.direction, tri.v[0], tri.v[1], tri.v[2], b_coord))
		return -1.0f; /* no collision */
	
	*normal = glm::normalize(glm::cross(tri.v[2]-tri.v[0], tri.v[1]-tri.v[0]));
	
	*intersect = getPointOnRay(r, b_coord.z);
	return b_coord.z;
}


/**
 * Test intersection between a ray and a mesh. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
	glm::vec3 *intersectionPoint, glm::vec3 *normal, bool *outside, const Triangle *tris)
{
	float t = -1.0f;
	float t_min = FLT_MAX;
	glm::vec3 tri_intersect_point;
	glm::vec3 tri_normal;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	bool hit = false;


#if BOUNDING_BOX
	/* do box-intersection with bounding boxes */
	printf("mincoords: (%f, %f, %f)\tmaxcoords: (%f, %f, %f)\n",
		mesh.mincoords.x, mesh.mincoords.y, mesh.mincoords.z,
		mesh.maxcoords.x, mesh.maxcoords.y, mesh.maxcoords.z);

	glm::vec3 d = glm::normalize(r.direction);
	auto max_of_mins = max(0.f,
		(mesh.mincoords.x - r.origin.x) / d.x,
		(mesh.mincoords.y - r.origin.y) / d.y,
		(mesh.mincoords.z - r.origin.z) / d.z);
	auto min_of_maxes = min(
		(mesh.maxcoords.x - r.origin.x) / d.x,
		(mesh.maxcoords.y - r.origin.y) / d.y,
		(mesh.maxcoords.z - r.origin.z) / d.z);
	
	if (min_of_maxes < max_of_mins)
		return -1.0f;
#endif

	for (size_t i = mesh.triangle_start; i < mesh.triangle_start + mesh.triangle_n; i++) {
		t = triangle_intersection_test(tris[i], r, &tmp_intersect, &tmp_normal);

		if (t > 0.0f && t_min > t) {
			t_min = t;
			tri_intersect_point = tmp_intersect;
			tri_normal = tmp_normal;
			//printf("triangle %lld is closer\n", i);
			hit = true;
		}

	}

	if (t_min > 0.0f && hit) {
		*intersectionPoint = tri_intersect_point;
	//	*normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tri_normal, 1.0f)));
		*normal = tri_normal;
		*outside = glm::dot(r.direction, *normal) < 0;
	//printf("normal: (%f, %f, %f) (%f, %f, %f) %d\n", normal->x, normal->y, normal->z, r.direction.x, r.direction.y, r.direction.z, int(*outside));
	//printf("intersect = (%f, %f, %f)\n", intersectionPoint->x, intersectionPoint->y, intersectionPoint->z);
		return t_min;
	}



	return -1.0f;
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
	glm::vec3 *intersectionPoint, glm::vec3 *normal, bool *outside)
{
	Ray q;
	q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; xyz++) {
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/
		{
			float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
			float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? 1.f : -1.f;
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
		*outside = true;
		if (tmin <= 0) {
			tmin = tmax;
			tmin_n = tmax_n;
			*outside = false;
		}
		*intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		*normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
		return glm::length(r.origin - *intersectionPoint);
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
	glm::vec3 *intersectionPoint, glm::vec3 *normal, bool *outside)
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
		return -1;

	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;

	float t = 0;
	if (t1 < 0 && t2 < 0) {
		return -1;
	} else if (t1 > 0 && t2 > 0) {
		t = min(t1, t2);
		*outside = true;
	} else {
		t = max(t1, t2);
		*outside = false;
	}

	glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

	*intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
	*normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
	if (!outside) {
		*normal = -*normal;
	}

	return glm::length(r.origin - *intersectionPoint);
}
