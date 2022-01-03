#pragma once

#include "intersections.h"


// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

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

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}



__host__ __device__
float schlick_approx(float n2_n1, float cos1)
{
	float R_0 = (1.0f - n2_n1) / (1.0f + n2_n1);
	R_0 = R_0 * R_0;
	float p = (1-cos1);
	float s = p * p;
	s *= s;
	s *= p;
	return R_0 + (1 - R_0) * s; 
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
	PathSegment &path_segment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng)
{
	Ray &ray = path_segment.ray;
	// reflection if prob is <reflect_prob, refraction if prob > refract_prob, diffuse otherwise
	thrust::uniform_real_distribution<float> u01(0, 1);
	float prob = u01(rng);
	float p = u01(rng);

	if (prob > 1 - m.hasRefractive) { /* refraction */
		//based on PBRT 8.2 and RayTracingInOneWeekend
		glm::vec3 u_dir = glm::normalize(ray.direction);

		bool outwards = glm::dot(u_dir, normal) > 0.0f; /* is this ray leaving the object? */
		float n2_n1 = outwards ? m.indexOfRefraction : (1.0f / m.indexOfRefraction); /* ratio of n2 to n1 */

		float cosine = min(-glm::dot(u_dir, normal), 1.0f);
		float sine = sqrt(1.0f - cosine * cosine);


		bool reflects = (n2_n1 * sine > 1.0f) /*sine of second angle >=1 implies internal reflection */
			|| (cosine > 0.0f && schlick_approx(n2_n1, cosine) > p);

		ray.direction = glm::normalize(reflects ? glm::reflect(u_dir, normal * (1.0f - 2 * outwards))
			: glm::refract(u_dir, normal * (1.0f - 2 * outwards), n2_n1));

		ray.origin = intersect + 0.001f * normal * (2 * reflects - 1.0f) * (1.0f - 2 * outwards);

		if (reflects)
			path_segment.color *= m.specular.color;
	} else {
		path_segment.color *= m.color; /* these are same for reflect/diffuse */
		ray.origin = intersect;
		ray.direction = prob < m.hasReflective ? glm::reflect(ray.direction, normal) /* reflection */
			: calculateRandomDirectionInHemisphere(normal, rng); /* diffuse*/
	}

}
