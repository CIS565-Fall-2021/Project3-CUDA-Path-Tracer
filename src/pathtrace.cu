#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
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
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
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

static Scene* hst_scene = NULL;
static glm::vec3* device_image = NULL;
static Geom* device_geoms = NULL;
static Material* device_materials = NULL;
static PathSegment* device_paths = NULL;
static ShadeableIntersection* device_intersections = NULL;
static ShadeableIntersection* device_intersections_first_bounce = NULL;

const bool STREAM_COMPACTION = true;
const bool SORT_MATERIALS = false;
const bool CACHE_FIRST_BOUNCE = true;
static bool cached = false;

const int DEPTH = 3;

std::vector<long> iterationTimes;
std::vector<long> computeIntersectionsTimes;
std::vector<long> shadeFakeMaterialTimes;
std::vector<long> sortMaterialsTimes;
std::vector<long> streamCompactionTimes;
std::vector<long> numPaths;

std::vector<long> firstComputeIntersectionsTimes;
std::vector<long> cacheWriteTimes;
std::vector<long> cacheReadTimes;

chrono::steady_clock::time_point iterationStart;
chrono::steady_clock::time_point iterationEnd;

chrono::steady_clock::time_point computeIntersectionsStart;
chrono::steady_clock::time_point computeIntersectionsEnd;

chrono::steady_clock::time_point shadeFakeMaterialStart;
chrono::steady_clock::time_point shadeFakeMaterialEnd;

chrono::steady_clock::time_point sortMaterialsStart;
chrono::steady_clock::time_point sortMaterialsEnd;

chrono::steady_clock::time_point streamCompactionStart;
chrono::steady_clock::time_point streamCompactionEnd;

chrono::steady_clock::time_point firstComputeIntersectionsStart;
chrono::steady_clock::time_point firstComputeIntersectionsEnd;

chrono::steady_clock::time_point cacheWriteStart;
chrono::steady_clock::time_point cacheWriteEnd;

chrono::steady_clock::time_point cacheReadStart;
chrono::steady_clock::time_point cacheReadEnd;

struct NoRemainingBounces {
    __host__ __device__
    bool operator()(const PathSegment& pathSegment) {
        return pathSegment.remainingBounces;
    }
};

struct MaterialSort {
    __host__ __device__
    bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2) {
        return i1.materialId > i2.materialId;
    }
};

void allocateTriangleMemory(Geom& geom) {
    int bytes;
    int startIndex = (pow(8, DEPTH) - 1) / 7;
    int endIndex = (pow(8, DEPTH + 1) - 1) / 7;
    
    for (int i = startIndex; i < endIndex; i++) {
        if ((*geom.host_tree)[i].numTriangles) {
            cudaMalloc(
                &(*geom.host_tree)[i].device_triangles,
                (*geom.host_tree)[i].host_triangles->size() * sizeof(Triangle));
            cudaMemcpy(
                (*geom.host_tree)[i].device_triangles,
                (*geom.host_tree)[i].host_triangles->data(),
                (*geom.host_tree)[i].host_triangles->size() * sizeof(Triangle),
                cudaMemcpyHostToDevice);

            bytes += (*geom.host_tree)[i].host_triangles->size() * sizeof(Triangle);
        }
    }

    std::cout << "Allocated triangle memory\t(" << bytes << " bytes)" << std::endl;
}

void allocateTreeMemory(Geom& geom) {
    cudaMalloc(&geom.device_tree,  geom.host_tree->size() * sizeof(Node));
    cudaMemcpy(geom.device_tree, geom.host_tree->data(), geom.host_tree->size() * sizeof(Node), cudaMemcpyHostToDevice);

    std::cout << "Allocated tree memory\t\t(" << geom.host_tree->size() * sizeof(Node) << " bytes)" << std::endl;

}

void pathtraceInit(Scene *scene) {

    //// https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__THREAD_gfd87d16d2bbf4bc41a892f3f75bac5e0.html#gfd87d16d2bbf4bc41a892f3f75bac5e0
    //size_t lim;
    //cudaThreadGetLimit(&lim, cudaLimitMallocHeapSize);
    //std::cout << "OLD MallocHeapSize " << lim << std::endl;

    //size_t new_lim = 74299064 * 4;
    //cudaThreadSetLimit(cudaLimitMallocHeapSize, new_lim);
    //cudaThreadGetLimit(&lim, cudaLimitMallocHeapSize);
    //std::cout << "NEW MallocHeapSize " << lim << std::endl;

    //cudaThreadGetLimit(&lim, cudaLimitStackSize);
    //std::cout << "OLD StackSize " << lim << std::endl;

    //new_lim = 1024 * 64;
    //cudaThreadSetLimit(cudaLimitStackSize, new_lim);
    //cudaThreadGetLimit(&lim, cudaLimitStackSize);
    //std::cout << "NEW StackSize " << lim << std::endl;

    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // For each pixel, store a
    //  - vec3 (the color)
    //  - ray (PathSegment)
    //  - intersection (ShadeableIntersection)
    cudaMalloc(&device_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(device_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&device_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&device_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(device_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (CACHE_FIRST_BOUNCE) {
        cudaMalloc(&device_intersections_first_bounce, pixelcount * sizeof(ShadeableIntersection));
    }

    for (int i = 0; i < scene->geoms.size(); i++) {
        if (scene->geoms[i].type == BB) {
            //cudaMalloc(
            //    &scene->geoms[i].device_triangles,
            //    scene->geoms[i].host_triangles->size() * sizeof(Triangle));
            //cudaMemcpy(
            //    scene->geoms[i].device_triangles,
            //    scene->geoms[i].host_triangles->data(),
            //    scene->geoms[i].host_triangles->size() * sizeof(Triangle),
            //    cudaMemcpyHostToDevice);

            allocateTriangleMemory(scene->geoms[i]);
            allocateTreeMemory(scene->geoms[i]);
        }
    }

    // Also copy over geometries (Geom) and materials (Material)
    cudaMalloc(&device_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(device_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&device_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(device_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cached = false;
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(device_image);  // no-op if device_image is null
    cudaFree(device_paths);
    cudaFree(device_geoms);
    cudaFree(device_materials);
    cudaFree(device_intersections);
    cudaFree(device_intersections_first_bounce);

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
__global__ void generateRaysFromCamera(Camera camera, int iter, int traceDepth, PathSegment* pathSegments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < camera.resolution.x && y < camera.resolution.y) {
        int index = x + (y * camera.resolution.x);
        // consider using shared memory here, since all of these operations are on global memory
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = camera.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(camera.view
            - camera.right * camera.pixelLength.x * ((float)x - (float)camera.resolution.x * 0.5f)
            - camera.up * camera.pixelLength.y * ((float)y - (float)camera.resolution.y * 0.5f));

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(
    int depth, int num_paths,
    PathSegment* pathSegments,
    Geom* geoms, int geoms_size,
    ShadeableIntersection* intersections
) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths && pathSegments[path_index].remainingBounces) {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        float t_min = FLT_MAX;

        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        
        int hit_geom_index = -1;
        bool outside;

        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++) {
            Geom& geom = geoms[i];

            if (geom.type == CUBE) {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            } else if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            } else if (geom.type == BB) {
                t = octreeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            // Compute the minimum t from the intersection tests to determine which
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t) {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
        } else {
            //The ray hits something
            intersections[path_index].outside = outside;
            intersections[path_index].intersectionPt = intersect_point;
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
__global__ void shadeFakeMaterial (
    int iter, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int currentDepth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths && pathSegments[idx].remainingBounces) {
        ShadeableIntersection intersection = shadeableIntersections[idx];

        if (intersection.t > 0.0f) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, currentDepth);
            Material material = materials[intersection.materialId];
            float gap = material.hasRefractive ? 0.0001f : -0.0001f;

            scatterRay(
                pathSegments[idx],
                getPointOnRay(pathSegments[idx].ray, intersection.t, gap),
                intersection.surfaceNormal,
                material, rng,
                shadeableIntersections[idx].outside);
        } else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths) {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// Wrapper for the __global__ call
// Sets up the kernel calls and does memory management
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int maxTraceDepth = hst_scene->state.traceDepth;
    const Camera &camera = hst_scene->state.camera;
    const int pixelcount = camera.resolution.x * camera.resolution.y;

    const dim3 blockSizeRayGen(8, 8);
    const dim3 gridSizeRayGen(
            (camera.resolution.x + blockSizeRayGen.x - 1) / blockSizeRayGen.x,
            (camera.resolution.y + blockSizeRayGen.y - 1) / blockSizeRayGen.y);

    const int blockSizeCompute = 128;

    // INITIALIZE device_paths (PathSegment structs)
    generateRaysFromCamera <<<gridSizeRayGen, blockSizeRayGen >>>(camera, iter, maxTraceDepth, device_paths);
    checkCUDAError("generate camera rays");

    int currentTraceDepth = 0;
    PathSegment* device_path_end = device_paths + pixelcount;
    int num_paths = device_path_end - device_paths;

    bool iterationComplete = false;

    while (!iterationComplete) {
        if (iter >= 30 && iter < 50) {
            if (iter == 30) {
                numPaths.push_back(num_paths);
            } else {
                numPaths[currentTraceDepth] += num_paths;
            }
        }

        iterationStart = chrono::high_resolution_clock::now();
        // RESET device_intersections (ShadeableIntersection structs)
        cudaMemset(device_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 gridSizePathSegmentTracing = (num_paths + blockSizeCompute - 1) / blockSizeCompute;

        computeIntersectionsStart = chrono::high_resolution_clock::now();
        if (currentTraceDepth == 0 && CACHE_FIRST_BOUNCE) {
            if (!cached) {
                cached = true;
                // POPULATE device_intersections (ShadeableIntersection structs)
                firstComputeIntersectionsStart = chrono::high_resolution_clock::now();
                computeIntersections << <gridSizePathSegmentTracing, blockSizeCompute >> > (
                    currentTraceDepth, num_paths, device_paths, device_geoms, hst_scene->geoms.size(), device_intersections);
                checkCUDAError("Error with computeIntersections()");
                cudaDeviceSynchronize();
                firstComputeIntersectionsEnd = chrono::high_resolution_clock::now();

                // POPULATE device_intersections_first_bounce (cache of ShadeableIntersection structs)
                cacheWriteStart = chrono::high_resolution_clock::now();
                cudaMemcpy(
                    device_intersections_first_bounce,
                    device_intersections,
                    pixelcount * sizeof(ShadeableIntersection),
                    cudaMemcpyDeviceToDevice);
                checkCUDAError("Error with populating cache");
                cudaDeviceSynchronize();
                cacheWriteEnd = chrono::high_resolution_clock::now();

            } else {
                // POPULATE device_intersections using cache
                cacheReadStart = chrono::high_resolution_clock::now();
                cudaMemcpy(
                    device_intersections,
                    device_intersections_first_bounce,
                    pixelcount * sizeof(ShadeableIntersection),
                    cudaMemcpyDeviceToDevice);
                checkCUDAError("Error with reading from cache");
                cudaDeviceSynchronize();
                cacheReadEnd = chrono::high_resolution_clock::now();
            }
        } else {
            // POPULATE device_intersections (ShadeableIntersection structs)
            computeIntersections << <gridSizePathSegmentTracing, blockSizeCompute >> > (
                currentTraceDepth, num_paths, device_paths, device_geoms, hst_scene->geoms.size(), device_intersections);
            checkCUDAError("Error with computeIntersections()");
            cudaDeviceSynchronize();
        }
        computeIntersectionsEnd = chrono::high_resolution_clock::now();
        
        if (SORT_MATERIALS) {
            sortMaterialsStart = chrono::high_resolution_clock::now();
            thrust::stable_sort_by_key(
                thrust::device,
                device_intersections,
                device_intersections + num_paths,
                device_paths,
                MaterialSort());
            sortMaterialsEnd = chrono::high_resolution_clock::now();
        }

        shadeFakeMaterialStart = chrono::high_resolution_clock::now();
        shadeFakeMaterial<<<gridSizePathSegmentTracing, blockSizeCompute>>> (
            iter, num_paths, device_intersections, device_paths, device_materials, currentTraceDepth);
        checkCUDAError("Error with shadeFakeMaterial()");
        cudaDeviceSynchronize();
        shadeFakeMaterialEnd = chrono::high_resolution_clock::now();

        if (STREAM_COMPACTION) {
            streamCompactionStart = chrono::high_resolution_clock::now();
            device_path_end = thrust::stable_partition(thrust::device, device_paths, device_path_end, NoRemainingBounces());
            num_paths = device_path_end - device_paths;
            streamCompactionEnd = chrono::high_resolution_clock::now();
        }

        iterationEnd = chrono::high_resolution_clock::now();
        if (iter >= 30 && iter < 50) {
            if (iter == 30) {
                iterationTimes.push_back(chrono::duration_cast<chrono::microseconds>(iterationEnd - iterationStart).count());
                computeIntersectionsTimes.push_back(chrono::duration_cast<chrono::microseconds>(computeIntersectionsEnd - computeIntersectionsStart).count());
                shadeFakeMaterialTimes.push_back(chrono::duration_cast<chrono::microseconds>(shadeFakeMaterialEnd - shadeFakeMaterialStart).count());
                sortMaterialsTimes.push_back(chrono::duration_cast<chrono::microseconds>(sortMaterialsEnd - sortMaterialsStart).count());
                streamCompactionTimes.push_back(chrono::duration_cast<chrono::microseconds>(streamCompactionEnd - streamCompactionStart).count());
            } else {
                iterationTimes[currentTraceDepth] += chrono::duration_cast<chrono::microseconds>(iterationEnd - iterationStart).count();
                computeIntersectionsTimes[currentTraceDepth] += chrono::duration_cast<chrono::microseconds>(computeIntersectionsEnd - computeIntersectionsStart).count();
                shadeFakeMaterialTimes[currentTraceDepth] += chrono::duration_cast<chrono::microseconds>(shadeFakeMaterialEnd - shadeFakeMaterialStart).count();
                sortMaterialsTimes[currentTraceDepth] += chrono::duration_cast<chrono::microseconds>(sortMaterialsEnd - sortMaterialsStart).count();
                streamCompactionTimes[currentTraceDepth] += chrono::duration_cast<chrono::microseconds>(streamCompactionEnd - streamCompactionStart).count();
            }
        }

        if (iter >= 30 && iter < 50 && currentTraceDepth == 0) {
            firstComputeIntersectionsTimes.push_back(chrono::duration_cast<chrono::microseconds>(firstComputeIntersectionsEnd - firstComputeIntersectionsStart).count());
            cacheReadTimes.push_back(chrono::duration_cast<chrono::microseconds>(cacheReadEnd - cacheReadStart).count());
            cacheWriteTimes.push_back(chrono::duration_cast<chrono::microseconds>(cacheWriteEnd - cacheWriteStart).count());
        }

        currentTraceDepth++;
        iterationComplete = num_paths == 0 || currentTraceDepth == maxTraceDepth;
    }



    if (iter == 51) {
        long total;

        std::cout << "firstComputeIntersectons" << std::endl;
        for (int i = 0; i < 20; i++) {
            std::cout << i << " " << firstComputeIntersectionsTimes[i] << std::endl;
            total += firstComputeIntersectionsTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;

        std::cout << "cacheRead" << std::endl;
        for (int i = 0; i < 20; i++) {
            std::cout << i << " " << cacheReadTimes[i] << std::endl;
            total += cacheReadTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;

        std::cout << "cacheWrite" << std::endl;
        for (int i = 0; i < 20; i++) {
            std::cout << i << " " << cacheWriteTimes[i] << std::endl;
            total += cacheWriteTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;


        std::cout << "numPaths" << std::endl;
        for (int i = 0; i < maxTraceDepth; i++) {
            std::cout << i << " " << numPaths[i]/20 << std::endl;
            total += numPaths[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;


        std::cout << "iteration times" << std::endl;
        for (int i = 0; i < maxTraceDepth; i++) {
            std::cout << i << " " << iterationTimes[i] / 20 << std::endl;
            total += iterationTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;


        std::cout << "computeIntersections times" << std::endl;
        for (int i = 0; i < maxTraceDepth; i++) {
            std::cout << i << " " << computeIntersectionsTimes[i] / 20 << std::endl;
            total += computeIntersectionsTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;


        std::cout << "shadeFakeMaterial times" << std::endl;
        for (int i = 0; i < maxTraceDepth; i++) {
            std::cout << i << " " << shadeFakeMaterialTimes[i] / 20 << std::endl;
            total += shadeFakeMaterialTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;


        std::cout << "sortMaterials times" << std::endl;
        for (int i = 0; i < maxTraceDepth; i++) {
            std::cout << i << " " << sortMaterialsTimes[i] / 20 << std::endl;
            total += sortMaterialsTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;


        std::cout << "streamCompaction times" << std::endl;
        for (int i = 0; i < maxTraceDepth; i++) {
            std::cout << i << " " << streamCompactionTimes[i] / 20 << std::endl;
            total += streamCompactionTimes[i];
        }
        std::cout << "TOTAL " << total / 20 << std::endl;
        total = 0;
    }

    // Assemble this iteration and apply it to the image
    dim3 gridSizePixels = (pixelcount + blockSizeCompute - 1) / blockSizeCompute;
    finalGather<<<gridSizePixels, blockSizeCompute >>>(pixelcount, device_image, device_paths);
    checkCUDAError("Error with finalGather()");
    cudaDeviceSynchronize();

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<gridSizeRayGen, blockSizeRayGen>>>(pbo, camera.resolution, iter, device_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), device_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
