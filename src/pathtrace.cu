#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/common.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_BOUNCE 0
#define PERFORMANCE_ANALYSIS 1

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

#if PERFORMANCE_ANALYSIS
const int numIters = 100;
static float totalTime = 0.f;
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}
#endif

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = nullptr;
static glm::vec3* dev_image = nullptr;
static Geom* dev_geoms = nullptr;
static Triangle* dev_triangles = nullptr;
static Material* dev_materials = nullptr;
static glm::vec3* dev_texData = nullptr;
static PathSegment* dev_paths = nullptr;
static ShadeableIntersection* dev_intersections = nullptr;
static ShadeableIntersection* dev_cachedIntersections = nullptr;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_cachedIntersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_cachedIntersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (scene->texData.size() > 0)
    {
        cudaMalloc(&dev_texData, scene->texData.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_texData, scene->texData.data(), scene->texData.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_triangles);
    cudaFree(dev_materials);
    cudaFree(dev_texData);
    cudaFree(dev_intersections);
    cudaFree(dev_cachedIntersections);

    checkCUDAError("pathtraceFree");
}

// Generate PathSegments with rays from the camera through the screen into the 
// scene, which is the first bounce of rays.
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) 
    {
        int index = x + (y * cam.resolution.x);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);

        Ray r;
        r.origin = cam.position;

#if (CACHE_FIRST_BOUNCE == 0)
        // stochastic sampled anti-aliasing
        thrust::uniform_real_distribution<float> offset(-0.5, 0.5);
        glm::vec2 point(x + offset(rng), y + offset(rng));
        r.direction = glm::normalize(cam.view
                                     - cam.right * cam.pixelLength.x * ((float)point.x - (float)cam.resolution.x * 0.5f)
                                     - cam.up * cam.pixelLength.y * ((float)point.y - (float)cam.resolution.y * 0.5f));
#else
        r.direction = glm::normalize(cam.view
                                     - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                                     - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
#endif

#if (CACHE_FIRST_BOUNCE == 0)
        // depth-of-field
        if (cam.aperture > 0)
        {
            thrust::uniform_real_distribution<float> u01(0, 1);

            glm::vec3 forward = glm::normalize(cam.lookAt - cam.position);
            glm::vec3 right = glm::normalize(glm::cross(forward, cam.up));
            glm::vec3 focalPoint = r.origin + cam.focalDist * r.direction;

            float angle = u01(rng) * 2.f * PI;
            float radius = cam.aperture * glm::sqrt(u01(rng));

            r.origin += radius * (cos(angle) * right + sin(angle) * cam.up);
            r.direction = glm::normalize(focalPoint - r.origin);
        }
#endif

        pathSegments[index].ray = r;
        pathSegments[index].color = glm::vec3(1.0f, 1.0f, 1.0f);
        pathSegments[index].pixelIndex = index;
        pathSegments[index].remainingBounces = traceDepth;
    }
}

// handles generating ray intersections.
__global__ void computeIntersections(int depth,  
                                     PathSegment* pathSegments, int num_paths,
                                     Geom* geoms, int geoms_size,
                                     Triangle* tris,
                                     Material* mats,
                                     glm::vec3* texData,
                                     ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

#if (STREAM_COMPACTION == 0)
        if (pathSegment.remainingBounces <= 0)
        {
            intersections[path_index].t = -1.f;
            return;
        }
#endif

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            if (geom.type == MESH)
            {
                if (aabbIntersectionTest(geom.aabb, pathSegment.ray))
                {
                    for (int j = geom.triBeginIdx; j < geom.triEndIdx; ++j)
                    {
                        t = triangleIntersectionTest(geom, tris[j], pathSegment.ray, mats[geom.materialid], texData,
                                                     tmp_intersect, tmp_normal, tmp_uv);
                        if (t > 0.f && t_min > t)
                        {
                            t_min = t;
                            hit_geom_index = i;
                            intersect_point = tmp_intersect;
                            normal = tmp_normal;
                            uv = tmp_uv;
                        }
                    }
                }
            }
            else 
            {
                if (geom.type == CUBE)
                {
                    t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }
                else if (geom.type == SPHERE)
                {
                    t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }
                if (t > 0.0f && t_min > t)
                {
                    t_min = t;
                    hit_geom_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                }
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
            intersections[path_index].uv = uv;
        }
    }
}

// processes rays based on intersections. 
// For non-terminating rays calls scatterRay for scattering and shading.
__global__ void shadeBSDF(int iter,
                          int depth,
                          int num_paths,
                          ShadeableIntersection* shadeableIntersections,
                          PathSegment* pathSegments,
                          Material* materials,
                          glm::vec3* dev_texData) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& pathSeg = pathSegments[idx];
    ShadeableIntersection& intersection = shadeableIntersections[idx];

    if (intersection.t > 0.f) 
    {
        Material mat = materials[intersection.materialId];
        if (mat.emittance > 0.f) 
        {
            pathSeg.remainingBounces = 0;
            pathSeg.color *= mat.color * mat.emittance;
        }
        else 
        {
            int bounces = --pathSeg.remainingBounces;
            if (bounces > 0) 
            {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
                scatterRay(pathSeg, 
                           getPointOnRay(pathSeg.ray, intersection.t), 
                           intersection.surfaceNormal, 
                           intersection.uv,
                           mat, 
                           dev_texData,
                           rng);
            }
            else 
            {
                pathSeg.color = glm::vec3(0.f);
            }
        }
    }
    else 
    {
#if (STREAM_COMPACTION == 0)
        if (pathSeg.remainingBounces > 0) 
        {
            pathSeg.remainingBounces = 0;
            pathSeg.color = glm::vec3(0.f);
        }
#else
        pathSeg.remainingBounces = 0;
        pathSeg.color = glm::vec3(0.f);
#endif
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) 
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d((cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
                               (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    // * Finally, add this iteration's results to the image. 

    // perform one iteration of path tracing

#if PERFORMANCE_ANALYSIS
    if (iter <= numIters)
    {
        timer().startCpuTimer();
    }
#endif

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);

    int depth = 0;
    PathSegment* dev_paths_end = dev_paths + pixelcount;
    int num_paths = dev_paths_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    while (num_paths > 0) 
    {
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
        if (depth == 0)
        {
            if (iter == 1)
            {
                computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>
                    (depth, dev_paths, num_paths, dev_geoms, hst_scene->geoms.size(), dev_triangles, dev_materials, dev_texData, dev_intersections);
                cudaMemcpy(dev_cachedIntersections, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
            else
            {
                cudaMemcpy(dev_intersections, dev_cachedIntersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
        }
        else
        {
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>
                (depth, dev_paths, num_paths, dev_geoms, hst_scene->geoms.size(), dev_triangles, dev_materials, dev_texData, dev_intersections);
        }
#else
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>
            (depth, dev_paths, num_paths, dev_geoms, hst_scene->geoms.size(), dev_triangles, dev_materials, dev_texData, dev_intersections);
#endif

        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.

#if SORT_BY_MATERIAL
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths);
#endif
        
        shadeBSDF<<<numblocksPathSegmentTracing, blockSize1d>>>(iter, depth, num_paths, dev_intersections, dev_paths, dev_materials, dev_texData);

#if STREAM_COMPACTION
        dev_paths_end = thrust::partition(thrust::device, dev_paths, dev_paths_end, pathRemains());
        num_paths = dev_paths_end - dev_paths;
#else
        if (depth >= hst_scene->state.traceDepth)
        {
            break;
        }
#endif
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

#if PERFORMANCE_ANALYSIS
    if (iter <= numIters) 
    {
        timer().endCpuTimer();
        totalTime += timer().getCpuElapsedTimeForPreviousOperation();
        if (iter == numIters)
        {
            cout << "Path-trace time for " << numIters << " iterations: " << totalTime << "ms" << endl;
        }
    }
#endif
}
