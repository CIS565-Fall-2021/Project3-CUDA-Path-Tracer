#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

// Sort the rays so that rays interacting with the same material are contiguous in memory before shading
#define MATERIAL_SORT 1

// Cache the first bounce intersections for re-use across all subsequent iterations
#define CACHE_FIRST_BOUNCE 1

// Apply 4x stochastic sampling and average
#define ANTIALIASING 0

// Direct lighting by taking a final ray directly to a random point on an emissive object
#define DIRECT_LIGHT 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution,
    int iter, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static ShadeableIntersection *dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
#if CACHE_FIRST_BOUNCE
static ShadeableIntersection *dev_first_intersections = NULL;
static PathSegment *dev_first_paths = NULL;
#endif

#if ANTIALIASING
static glm::vec3 *dev_final_image = NULL;
#endif

#if DIRECT_LIGHT
static int num_lights;
static int *dev_lights = NULL;
#endif

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;

#if ANTIALIASING
    const int pixelcount = 4 * cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_final_image, pixelcount / 4 * sizeof(glm::vec3));
    cudaMemset(dev_final_image, 0, pixelcount / 4 * sizeof(glm::vec3));
#else
    const int pixelcount = cam.resolution.x * cam.resolution.y;
#endif

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
#if CACHE_FIRST_BOUNCE
    cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_first_paths, pixelcount * sizeof(PathSegment));
#endif

#if DIRECT_LIGHT
    // Add emissive geoms as lights
    num_lights = 0;
    std::vector<int> lights;
    for (int i = 0; i < scene->geoms.size(); i++) {
        Geom &geom = scene->geoms[i];
        if (scene->materials[geom.materialid].emittance > 0.f) {
            lights.push_back(i);
            num_lights++;
        }
    }
    cudaMalloc(&dev_lights, num_lights * sizeof(int));
    cudaMemcpy(dev_lights, lights.data(), num_lights * sizeof(int), cudaMemcpyHostToDevice);
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
#if CACHE_FIRST_BOUNCE
    cudaFree(dev_first_intersections);
    cudaFree(dev_first_paths);
#endif

#if ANTIALIASING
    cudaFree(dev_final_image);
#endif

#if DIRECT_LIGHT
    cudaFree(dev_lights);
#endif

    checkCUDAError("pathtraceFree");
}

// Sample point on disk
// Used for depth of field
__host__ __device__ void concentric_sample_disk(float *dx, float *dy, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u02(0, 1);
    float sx = 2 * u01(rng) - 1;
    float sy = 2 * u02(rng) - 1;

    // Trick referenced from http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
    float r;
    float theta;
    if (sx * sx > sy * sy) {
        r = sx;
        theta = (PI / 4.f) * (sy / sx);
    }
    else {
        r = sy;
        theta = (PI / 2.f) - (PI / 4.f) * (sx / sy);
    }

    *dx = r * cosf(theta);
    *dy = r * sinf(theta);
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment *pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

#if ANTIALIASING
    if (x < cam.resolution.x * 2 && y < cam.resolution.y * 2) {
        int index = x + (y * cam.resolution.x * 2);
        PathSegment &segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        thrust::uniform_real_distribution<float> u02(0, 1);

        // Add stochastic offset inside one pixel
        segment.ray.origin = cam.position;
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x / 2.f * ((float)x + u01(rng) - 0.5f - (float)cam.resolution.x * 2 * 0.5f)
            - cam.up * cam.pixelLength.y / 2.f * ((float)y + u02(rng) - 0.5f - (float)cam.resolution.y * 2 * 0.5f)
        );

        // Depth of field
        if (cam.lens_radius) {
            // Sample on lens
            float sample_x;
            float sample_y;
            concentric_sample_disk(&sample_x, &sample_y, rng);
            sample_x *= cam.lens_radius;
            sample_y *= cam.lens_radius;
            glm::vec3 len_pos(sample_x, sample_y, 0.f);

            // Update ray direction
            float ft = cam.focal_length / -segment.ray.direction.z;
            glm::vec3 focus_point = ft * segment.ray.direction;

            // Convert to world space
            segment.ray.origin += len_pos;
            segment.ray.direction = glm::normalize(focus_point - len_pos);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
#else
    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment &segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.origin = cam.position;
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        // Depth of field
        if (cam.lens_radius) {
            // Sample on lens
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            float sample_x;
            float sample_y;
            concentric_sample_disk(&sample_x, &sample_y, rng);
            sample_x *= cam.lens_radius;
            sample_y *= cam.lens_radius;
            glm::vec3 len_pos(sample_x, sample_y, 0.f);

            // Update ray direction
            float ft = cam.focal_length / -segment.ray.direction.z;
            glm::vec3 focus_point = ft * segment.ray.direction;

            // Convert to world space
            segment.ray.origin += len_pos;
            segment.ray.direction = glm::normalize(focus_point - len_pos);
        }
        
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
#endif
}

// Copy from utilities.cpp
__host__ __device__
glm::mat4 build_transformation_matrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
    glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment *pathSegments
    , Geom *geoms
    , int geoms_size
    , ShadeableIntersection *intersections
    , float time
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom &geom = geoms[i];

            // Blur moving objects
            if (geom.end_translation != geom.translation) {  

                geom.old_transform = geom.transform;
                geom.old_inverseTransform = geom.inverseTransform;
                geom.old_invTranspose = geom.invTranspose;

                // Linear interpolate current translation
                geom.transform = build_transformation_matrix(geom.translation + time * (geom.end_translation - geom.translation), geom.rotation, geom.scale);
                geom.inverseTransform = glm::inverse(geom.transform);
                geom.invTranspose = glm::inverseTranspose(geom.transform);
            }

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Set original translation back
            if (geom.end_translation != geom.translation) {
                geom.transform = geom.old_transform;
                geom.inverseTransform = geom.old_inverseTransform;
                geom.invTranspose = geom.old_invTranspose;
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter
    , int num_paths
    , ShadeableIntersection *shadeableIntersections
    , PathSegment *pathSegments
    , Material *materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                // Terminate when hitting a light
                pathSegments[idx].remainingBounces = 0;

                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                // Accumulate current color and generate bounce ray
                scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            // Terminate if nothing hitted
            pathSegments[idx].remainingBounces = 0;

            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__host__ __device__ glm::vec3 sample_cube(Geom cube, thrust::default_random_engine &rng, float* pdf) {
    // Sample in unit cube
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u02(0, 1);
    thrust::uniform_real_distribution<float> u03(0, 1);

    // Convert to world space
    glm::vec3 sample_point = multiplyMV(cube.transform, glm::vec4(u01(rng) - .5f, u02(rng) - .5f, u03(rng) - .5f, 1.f));
    // Inverse of volume
    *pdf = 1.f / (cube.scale.x * cube.scale.y * cube.scale.z);
    return sample_point;
}

__global__ void shade_direct_light(
    int iter
    , int num_paths
    , ShadeableIntersection *shadeableIntersections
    , PathSegment *pathSegments
    , Material *materials
    , Geom *geoms
    , int geoms_size
    , int *lights
    , int light_num
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment &pathSegment = pathSegments[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...

            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            
            glm::vec3 intersection_pos = getPointOnRay(pathSegment.ray, intersection.t);

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                // Terminate when hitting a light
                pathSegment.remainingBounces = 0;

                pathSegment.color *= (materialColor * material.emittance);
            }
            // Bounce on reflective or refractive surface
            else if (material.hasReflective || material.hasRefractive) {               
                // Accumulate current color and generate bounce ray
                scatterRay(pathSegment, intersection_pos, intersection.surfaceNormal, material, rng);
            }
            // For diffuse surface, take a final ray directly to a random point on an emissive object
            else {
                // Shadow ray does not reach lights
                if (pathSegment.remainingBounces == 1) {
                    pathSegment.remainingBounces--;
                    pathSegment.color *= glm::vec3(0.f, 0.f, 0.f);
                    return;
                }
                // Shoot final ray to light when hitting a diffuse surface
                pathSegment.remainingBounces = 1;

                // Choose a light randomly
                int light_index = int(u01(rng) * light_num);
                Geom &light = geoms[lights[light_index]];

                // Sample on light
                float pdf;
                glm::vec3 sample_light = sample_cube(light, rng, &pdf);
                // Attenuate light by square of distance
                float distance = glm::length2(sample_light - intersection_pos);

                // Shoot shadow ray
                pathSegment.ray.origin = intersection_pos;
                pathSegment.ray.direction = glm::normalize(sample_light - intersection_pos);   

                // Shade by incident angle
                pathSegment.color *= (materialColor * abs(glm::dot(pathSegment.ray.direction, intersection.surfaceNormal)) / (distance * pdf));
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            // Terminate if nothing hitted
            pathSegment.remainingBounces = 0;

            pathSegment.color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image, PathSegment *iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}


// Perform 4x downsampling on image
__global__ void down_sample(glm::ivec2 resolution, glm::vec3 *image, glm::vec3 *final_image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int out_index = x + (y * resolution.x);
        int in_index = 2 * x + (2 * y * resolution.x * 2);
        final_image[out_index] = (image[in_index] + image[in_index + 1] + image[in_index + resolution.x * 2] + image[in_index + resolution.x * 2 + 1]) / 4.f;
    }
}

// Determine whether to terminate by remaining bounce
// Used for thrust::stable_partition
struct is_path_terminated
{
    __host__ __device__
        bool operator()(const PathSegment &seg)
    {
        return seg.remainingBounces;
    }
};

// Compare by material id
// Used for thrust::sort_by_key
struct compare_material
{
    __host__ __device__
        bool operator()(const ShadeableIntersection &inter1, const ShadeableIntersection &inter2)
    {
        return inter1.materialId < inter2.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;

#if ANTIALIASING
    const int pixelcount = 4 * cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(16, 16);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x * 2 + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y * 2 + blockSize2d.y - 1) / blockSize2d.y);

    // 2D block for downsample
    const dim3 half_blockSize2d(8, 8);
    const dim3 half_blocksPerGrid2d(
        (cam.resolution.x + half_blockSize2d.x - 1) / half_blockSize2d.x,
        (cam.resolution.y + half_blockSize2d.y - 1) / half_blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 256;
#else
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;
#endif

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

#if CACHE_FIRST_BOUNCE
    if (iter == 1) {
        generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
        checkCUDAError("generate camera ray");

        // Cache in first iteration
        cudaMemcpy(dev_first_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
    }
    else {
        // Use cache in subsequent iterations
        cudaMemcpy(dev_paths, dev_first_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
    }
#else
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
#endif

    int depth = 0;
    PathSegment *dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    // Take random time elapsed ratio
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, 0, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float time = u01(rng);

    bool iterationComplete = false;
    while (!iterationComplete) {

#if CACHE_FIRST_BOUNCE
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        // Use cache in subsequent iterations
        if (iter != 1 && depth == 0) {
            cudaMemcpy(dev_intersections, dev_first_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else {
            // clean shading chunks
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

            // tracing
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                , time
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();

            // Cache in first iteration
            if (depth == 0) {
                cudaMemcpy(dev_first_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
        }
#else
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            , time
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
#endif
        
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
      // evaluating the BSDF.
      // Start off with just a big kernel that handles all the different
      // materials you have in the scenefile.
      // TODO: compare between directly shading the path segments and shading
      // path segments that have been reshuffled to be contiguous in memory.

#if MATERIAL_SORT
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_material());
#endif

#if DIRECT_LIGHT
        shade_direct_light << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lights,
            num_lights
            );
#else
        shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );
#endif

        // Stream compaction
        dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, is_path_terminated());
        num_paths = dev_path_end - dev_paths;

        // End if all paths terminate
        if (num_paths == 0) {
            iterationComplete = true; // TODO: should be based off stream compaction results.
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

#if ANTIALIASING
    // Average adjacent colors
    down_sample << <half_blocksPerGrid2d, half_blockSize2d >> > (cam.resolution, dev_image, dev_final_image);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <half_blocksPerGrid2d, half_blockSize2d >> > (pbo, cam.resolution, iter, dev_final_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_final_image,
        pixelcount / 4 * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
#else
    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
#endif     
}
