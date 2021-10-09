#pragma once

#include "intersections.h"

#define STRATIFIED_SAMPLING 1
#define HALTON_SAMPLING 0

//Use for Stratified sampling: division number on each side
#define SAMPLE_DIVISION 10

//Use for Halton sampling, 100 gives clear patterns on the render, 1000 is ok
#define SEQUENCE_LENGTH 1000


//Halton sequence: en.wikipedia.org/wiki/Halton_sequence
__host__ __device__ float halton(int base, int index) {
    float f = 1.f;
    float r = 0.f;
    while (index > 0) {
        f = f / base;
        r = r + f * (index % base);
        index = (int) index / base;
    }
    return r;
}

//2 > d:\a_gpu565\project3 - cuda - path - tracer\src\interactions.h(21) : warning: calling a __host__ function("double  ::floor<int, (int)0> (T1)") from a __host__ __device__ function("halton") is not allowed


// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u02(0, 1);
    thrust::uniform_real_distribution<float> u03(0, 1);
    float r1 = u01(rng);
    float r2 = u02(rng);

#if STRATIFIED_SAMPLING
    // A number useful for scaling a square of size sqrtVal x sqrtVal to 1 x 1
    float invSqrtVal = 1.f / SAMPLE_DIVISION;
    int numSamples = SAMPLE_DIVISION * SAMPLE_DIVISION;

    // Getting uniform x, y coords
    int i = u01(rng) * numSamples;
    int y = i / SAMPLE_DIVISION;
    int x = i % SAMPLE_DIVISION;

    // Jitter the x, y coords
    glm::vec2 sample = glm::vec2((x + r1) * invSqrtVal,
        (y + r2) * invSqrtVal);
    r1 = sample.x;
    r2 = sample.y;

#elif HALTON_SAMPLING
    int i = u03(rng) * SEQUENCE_LENGTH;
    r1 = halton(2, i);
    r2 = halton(3, i);

#endif

    float up = sqrt(r1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = r2 * TWO_PI;

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
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

    //return if there's no bounces left
    if (pathSegment.remainingBounces == 0) {
        return;
    }
    glm::vec3 new_ray;
    thrust::uniform_real_distribution<float> u01(0, 1);

    //Handle perfectly diffuse
    if (!m.hasReflective && !m.hasReflective) {
        pathSegment.color *= m.color;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
    } 
    //Both reflection and refraction
    else if (m.hasRefractive && m.hasReflective) {
        glm::vec3 incident = pathSegment.ray.direction;

        float ior_in{ 0.f };
        float ior_out{ 0.f };

        float cosThetaI = glm::dot(incident, normal);
        //Determine whether inside or outside the surface
        //If cosThetaI is negative: shooting from outside
        if (cosThetaI > 0.f) {
            normal = -normal;
            ior_in = 1.f;
            ior_out = m.indexOfRefraction;
        }
        else {
            ior_in = m.indexOfRefraction;
            ior_out = 1.f;
        }

        //Randomly choose between reflection and refraction
        thrust::uniform_real_distribution<float> u01(0, 1);
        float R_0 = powf((ior_in - ior_out) / (ior_in + ior_out), 2.f);
        float ref_coeff = R_0 + (1 - R_0) * powf(1.f + glm::dot(normal, incident), 5.f);

        float index = ior_in / ior_out;

        if (u01(rng) < ref_coeff) {
            //This is reflection
            pathSegment.ray.direction = glm::reflect(incident, normal);
            pathSegment.ray.origin = intersect;
            pathSegment.color *= m.specular.color;
        }
        else {
            glm::vec3 refract_dir = glm::refract(incident, normal, 1.f / index);

            if (glm::length(refract_dir) == 0.f) {
                //This is total internal reflection
                pathSegment.ray.direction = glm::reflect(incident, normal);
                pathSegment.ray.origin = intersect;
                pathSegment.color *= m.color;
            }
            else {
                //This is refraction
                pathSegment.ray.direction = glm::normalize(refract_dir);
                pathSegment.ray.origin = intersect + 0.002f * pathSegment.ray.direction;
                pathSegment.color *= m.specular.color;
            }
        }
    }
    //Only reflection
    else {
        pathSegment.color *= m.specular.color;
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect;
    }
    
    pathSegment.remainingBounces--;
}
