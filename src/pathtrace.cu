#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <chrono>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
    thrust::default_random_engine
    makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution,
                               int iter, glm::vec3 *image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
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
static ShadeableIntersection *dev_intersectionCache = NULL;
static glm::vec3 *dev_baseColor = NULL;
static glm::vec3 *dev_aoRoughMetal = NULL;
static glm::vec3 *dev_normals = NULL;
// ...

void pathtraceInit(Scene *scene)
{
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

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
    cudaMalloc(&dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersectionCache, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_baseColor, scene->baseColorVec.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_baseColor, scene->baseColorVec.data(), scene->baseColorVec.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image); // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_intersectionCache);
    cudaFree(dev_baseColor);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment *pathSegments, int depth)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y)
    {

        int index = x + (y * cam.resolution.x);
        PathSegment &segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(
            cam.view -
            cam.right * cam.pixelLength.x * ((float)x
#ifdef ANTIALIASING
                                             + u01(rng) - 0.5f
#endif
                                             - (float)cam.resolution.x * 0.5f) -
            cam.up * cam.pixelLength.y * ((float)y
#ifdef ANTIALIASING
                                          + u01(rng) - 0.5f
#endif
                                          - (float)cam.resolution.y * 0.5f));

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth, int num_paths, PathSegment *pathSegments, Geom *geoms, int geoms_size, ShadeableIntersection *intersections)
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
        glm::vec2 tmp_uv;
        glm::vec2 uv;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom &geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_uv = glm::vec2(-1);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_uv = glm::vec2(-1);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            // else if (geom.type == MESH)
            // {
            //     t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            // }
            else if (geom.type == TRIANGLE)
            {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = geom.type == TRIANGLE ? tmp_uv : glm::vec2(-1);
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
            intersections[path_index].uvs = uv;
        }
    }
}

/**
 * @brief Shader that DOES do a BSDF Evaluation
 * 
 * @param iter iteration number
 * @param num_paths total number of paths/rays
 * @param shadeableIntersections array of shadeable intersections
 * @param pathSegments array of pathsegments
 * @param materials array of materials
 */
__global__ void shadeRealMaterial(
    int iter, int num_paths, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials, int depth, glm::vec3 *baseColor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
        return;
    if (pathSegments[idx].remainingBounces <= 0)
        return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f)
    {
        Material material = materials[intersection.materialId];
        if (material.emittance > 0.f) // case thing is light
        {
            glm::vec3 materialColor = material.color;
            pathSegments[idx].color *= (materialColor * material.emittance);
            pathSegments[idx].remainingBounces = 0;
        }
        else // case thing isnt light so calculate
        {
#ifdef DEBUG_SURFACE_NORMAL
            pathSegments[idx].color = intersection.surfaceNormal; // Debug only
            pathSegments[idx].remainingBounces = 0;
#elif defined(DEBUG_T_VAL)
            pathSegments[idx].color = glm::vec3(intersection.t * 0.01); // Debug only
            pathSegments[idx].remainingBounces = 0;
#elif defined(DEBUG_TEX_BASE_COLOR)
            // height lines of width pixels ->
            //    u * width, v * height, floor both
            //    nU, nV -> nU + width * nV
            int tmpidx = (int)(glm::floor(intersection.uvs.x * 4096) + 4096 * floor(intersection.uvs.y * 4096));
            pathSegments[idx].color = tmpidx < (4096 * 4096) &&
                                              intersection.uvs.x >= 0 &&
                                              intersection.uvs.y >= 0
                                          ? baseColor[tmpidx] / 255.f
                                          : intersection.surfaceNormal;
            //   : material.color; // Debug only
            pathSegments[idx].remainingBounces = 0;
#else
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
            pathSegments[idx].remainingBounces--;
#endif
        }
    }
    else // intersection't => black
    {
        pathSegments[idx].color = glm::vec3(0.0f);
        pathSegments[idx].remainingBounces = 0;
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

struct orderMaterials
{
    __host__ __device__ bool operator()(ShadeableIntersection const &a, ShadeableIntersection const &b)
    {
        return a.materialId < b.materialId;
    }
};
struct isDeadYet
{
    __host__ __device__ bool operator()(PathSegment const ps)
    {
        return ps.remainingBounces > 0;
    }
};
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter)
{
#ifdef TIME_PATHTRACE
    static double timerAcc = 0.0;
#endif
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

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
    int depth = 0;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, depth);
    checkCUDAError("generate camera ray");
    cudaDeviceSynchronize();
    checkCUDAError("device synch");

    PathSegment *dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
#ifdef TIME_PATHTRACE
    using TimerClass = std::chrono::high_resolution_clock::time_point;
    TimerClass start = std::chrono::high_resolution_clock::now();
#endif
#ifdef GROUP_RAYS
    thrust::device_ptr<PathSegment> device_t_paths(dev_paths);
    thrust::device_ptr<ShadeableIntersection> device_t_intersections(dev_intersections);
#endif

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // cout << "Loop start cpu noncuda" << endl;
        checkCUDAError("startOfLoop");
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        // cudaDeviceSynchronize();
        checkCUDAError("memset");
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#ifdef CACHE_FIRST
        if (depth == 0) // main increments iteration before calling
        {
            if (iter == 1)
            {
                computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                    depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersectionCache);
                checkCUDAError("trace first bounce");
                cudaDeviceSynchronize();
            }
            cudaMemcpy(dev_intersections, dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            // depth++;
        }
        else
#endif
        {
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }

#ifdef GROUP_RAYS
        thrust::sort_by_key(device_t_intersections, device_t_intersections + num_paths, device_t_paths, orderMaterials());
#endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        // shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(iter, num_paths, dev_intersections, dev_paths, dev_materials);
        shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(iter, num_paths, dev_intersections, dev_paths, dev_materials, depth, dev_baseColor);
        checkCUDAError("my shader");
        depth++;
#ifdef COMPACT_RAYS
        PathSegment *newEnd = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isDeadYet());
        num_paths = newEnd - dev_paths;
        if (num_paths < 1)
        {
            depth++;
        }
#endif
        iterationComplete =
            (depth >= traceDepth)
#ifdef COMPACT_RAYS
            || (num_paths <= 0)
#endif
            ; // TODO: should be based off stream compaction results.
    }
    num_paths = dev_path_end - dev_paths;

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

#ifdef TIME_PATHTRACE
    cudaDeviceSynchronize();
    TimerClass finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed(finish - start);
    double fElapsed = static_cast<decltype(fElapsed)>(elapsed.count());
    timerAcc += fElapsed;
#endif
#ifdef TIME_PATHTRACE
    if (iter >= 999)
        std::cout << "elapsed time: " << timerAcc << "miliseconds" << std::endl;
#endif

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
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
    int iter, int num_paths, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        { // if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f)
            {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else
            {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else
        {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}
