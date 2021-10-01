#pragma once

#include "intersections.h"


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}
// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
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

/*
__host__ __device__ void EmitterBSDF(PathSegment & pathSegment,
									glm::vec3 intersect,
									glm::vec3 normal,
									const Material &m,
									thrust::default_random_engine &rng,
                                    thrust::uniform_int_distribution<int> u01) {
	pathSegments[idx].color *= (materialColor * material.emittance);
	pathSegments[idx].remainingBounces = 0;
}

__host__ __device__ void DiffuseBSDF(int num_paths,
                                    PathSegment & pathSegment,
									glm::vec3 intersect,
									glm::vec3 normal,
									const Material &m,
									thrust::default_random_engine &rng,
                                    thrust::uniform_int_distribution<int> u01) {
    int index = (blockIdx.x * blockDimx.x) + threadIdx.x;
    if (index < num_paths) {
    }
}

__host__ __device__ void ReflectBSDF(PathSegment & pathSegment,
									glm::vec3 intersect,
									glm::vec3 normal,
									const Material &m,
									thrust::default_random_engine &rng, 
                                    thrust::uniform_int_distribution<int> u01) {
    if (u01(rng)) {
        pathSegment.ray.origin = intersect;
        // Thanks stack exchange: https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
        // reflectedVector = ray - 2 * normal * dot(ray, normal)
        pathSegment.ray.direction = pathSegment.ray.direction - 
            (2.0f * normal * glm::dot(pathSegment.ray.direction, normal));
    }
    else {
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    }
    // divide by the probability of either path because at each step both would
    // normally be sampled (but only one can be when pathtracing)
	pathSegment.color /= 0.5f;
}
*/

__host__ __device__ void scatterRay(PathSegment & pathSegment,
									glm::vec3 intersect,
									glm::vec3 normal,
									const Material &m,
									thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_int_distribution<int> u01(0, 1);
    if (m.hasReflective && u01(rng)) {
        pathSegment.ray.origin = intersect;
        // Thanks stack exchange: https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
        // reflectedVector = ray - 2 * normal * dot(ray, normal)
        pathSegment.ray.direction = pathSegment.ray.direction - 
            (2.0f * normal * glm::dot(pathSegment.ray.direction, normal));
    }
    else {
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    }
    if (m.hasReflective) {
        pathSegment.color /= 0.5f;
    }
}


// --- Shaders ---
/*

// allShader has conditionals for all BSDFs. It's inefficient, but it gets us stared 
__global__ void shadeDiffuse(int iter,
							 int num_paths,
							 ShadeableIntersection * shadeableIntersections,
							 PathSegment * pathSegments,
							 Material * materials){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths){
		ShadeableIntersection intersection = shadeableIntersections[idx];
        // Set up the RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        pathSegments[idx].color *= materialColor;
        pathSegments[idx].remainingBounces--;
        pathSegments[idx].ray.origin =  getPointOnRay(pathSegments[idx].ray, intersection.t);
        pathSegments[idx].ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
  }
}

__global__ void shadeEmitter(int iter,
                             //int startIndex,
							 int num_paths,
							 ShadeableIntersection * shadeableIntersections,
							 PathSegment * pathSegments,
							 Material * materials){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths){
        
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > -1.0f) {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            pathSegments[idx].color *= (materialColor * material.emittance);
            pathSegments[idx].remainingBounces = 0;
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}
*/
