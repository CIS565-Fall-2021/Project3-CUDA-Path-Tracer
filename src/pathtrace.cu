// #include <cstdio>
// #include <cuda.h>
// #include <cmath>
// #include <thrust/execution_policy.h>
// #include <thrust/random.h>
// #include <thrust/remove.h>

// #include "sceneStructs.h"
// #include "scene.h"
// #include "glm/glm.hpp"
// #include "glm/gtx/norm.hpp"
// #include "utilities.h"
// #include "pathtrace.h"
// #include "intersections.h"
// #include "interactions.h"

// #define ERRORCHECK 1

// #define ANTI_ALIASING 0
// #define CACHE_BOUNCE 0
// #define SORT_MATERIALS 0
// #define DEPTH_OF_FIELD 0
// #define DIRECT_LIGHTING 1

// #define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// #define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
// void checkCUDAErrorFn(const char *msg, const char *file, int line) {
// #if ERRORCHECK
//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     if (cudaSuccess == err) {
//         return;
//     }

//     fprintf(stderr, "CUDA error");
//     if (file) {
//         fprintf(stderr, " (%s:%d)", file, line);
//     }
//     fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
// #  ifdef _WIN32
//     getchar();
// #  endif
//     exit(EXIT_FAILURE);
// #endif
// }

// __host__ __device__
// thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
//     int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
//     return thrust::default_random_engine(h);
// }

// //Kernel that writes the image to the OpenGL PBO directly.
// __global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
//         int iter, glm::vec3* image) {
//     int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//     int y = (blockIdx.y * blockDim.y) + threadIdx.y;

//     if (x < resolution.x && y < resolution.y) {
//         int index = x + (y * resolution.x);
//         glm::vec3 pix = image[index];

//         glm::ivec3 color;
//         color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
//         color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
//         color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

//         // Each thread writes one pixel location in the texture (textel)
//         pbo[index].w = 0;
//         pbo[index].x = color.x;
//         pbo[index].y = color.y;
//         pbo[index].z = color.z;
//     }
//     // checkCUDAError("sendImageToPBO");
// }

// static Scene * hst_scene = NULL;
// static glm::vec3 * dev_image = NULL;
// static Geom * dev_geoms = NULL;
// static Material * dev_materials = NULL;
// static PathSegment * dev_paths = NULL;
// static ShadeableIntersection * dev_intersections = NULL;
// static ShadeableIntersection* dev_intersection_first_bounce = NULL;
// #if DIRECT_LIGHTING
// static Geom* dev_lights = NULL;
// #endif
// // TODO: static variables for device memory, any extra info you need, etc
// // ...

// void pathtraceInit(Scene *scene) {
//     hst_scene = scene;
//     const Camera &cam = hst_scene->state.camera;
//     const int pixelcount = cam.resolution.x * cam.resolution.y;

//     cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
//     cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

//   	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

//   	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
//   	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

//   	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
//   	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

//   	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
//   	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
//     // TODO: initialize any extra device memeory you need
// #if CACHE_BOUNCE || SORT_MATERIALS
//     cudaMalloc(&dev_first_bounce, pixelcount * sizeof(ShadeableIntersection));
//     cudaMemset(dev_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));
// #endif

// #if DIRECT_LIGHTING
//     cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
//     cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
// #endif

//     checkCUDAError("pathtraceInit");

// }

// void pathtraceFree() {
//     cudaFree(dev_image);  // no-op if dev_image is null
//   	cudaFree(dev_paths);
//   	cudaFree(dev_geoms);
//   	cudaFree(dev_materials);
//   	cudaFree(dev_intersections);
//     // TODO: clean up any extra device memory you created

//     checkCUDAError("pathtraceFree");
// }


// __host__ __device__
// glm::vec3 pointOnPlane(Geom light, thrust::default_random_engine& rng) {
//     thrust::uniform_real_distribution<float> u01(0, 1);
//     glm::vec2 pt(u01(rng), u01(rng));
//     glm::vec3 planePt = glm::vec3((pt - glm::vec2(0.5f)), 0.f);
//     return glm::vec3(light.transform * glm::vec4(planePt, 1.f));
// }
// /**
// * Generate PathSegments with rays from the camera through the screen into the
// * scene, which is the first bounce of rays.
// *
// * Antialiasing - add rays for sub-pixel sampling
// * motion blur - jitter rays "in time"
// * lens effect - jitter ray origin positions based on a lens
// */
// __global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
// {
// 	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
// 	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

// 	if (x < cam.resolution.x && y < cam.resolution.y) {
// 		int index = x + (y * cam.resolution.x);
// 		PathSegment & segment = pathSegments[index];

// 		segment.ray.origin = cam.position;
//     segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

// 		// TODO: implement antialiasing by jittering the ray
// 		segment.ray.direction = glm::normalize(cam.view
// 			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
// 			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
// 			);

// 		segment.pixelIndex = index;
// 		segment.remainingBounces = traceDepth;
// 	}
// }

// // TODO:
// // computeIntersections handles generating ray intersections ONLY.
// // Generating new rays is handled in your shader(s).
// // Feel free to modify the code below.
// __global__ void computeIntersections(int depth, int num_paths, PathSegment * pathSegments, 
// Geom * geoms, int geoms_size, ShadeableIntersection * intersections)
// {
// 	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (path_index < num_paths)
// 	{
// 		PathSegment pathSegment = pathSegments[path_index];

// 		float t;
// 		glm::vec3 intersect_point;
// 		glm::vec3 normal;
// 		float t_min = FLT_MAX;
// 		int hit_geom_index = -1;
// 		bool outside = true;

// 		glm::vec3 tmp_intersect;
// 		glm::vec3 tmp_normal;

// 		// naive parse through global geoms

// 		for (int i = 0; i < geoms_size; i++)
// 		{
// 			Geom & geom = geoms[i];

// 			if (geom.type == CUBE)
// 			{
// 				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
// 			}
// 			else if (geom.type == SPHERE)
// 			{
// 				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
// 			}
// 			// TODO: add more intersection tests here... triangle? metaball? CSG?
//       else if (geom.type == TRIANGLE){
//         t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
//       }
// 			// Compute the minimum t from the intersection tests to determine what
// 			// scene geometry object was hit first.
// 			if (t > 0.0f && t_min > t)
// 			{
// 				t_min = t;
// 				hit_geom_index = i;
// 				intersect_point = tmp_intersect;
// 				normal = tmp_normal;
// 			}
// 		}

// 		if (hit_geom_index == -1)
// 		{
// 			intersections[path_index].t = -1.0f;
// 		}
// 		else
// 		{
// 			//The ray hits something
// 			intersections[path_index].t = t_min;
// 			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
// 			intersections[path_index].surfaceNormal = normal;
// 		}
// 	}
// }

// // LOOK: "fake" shader demonstrating what you might do with the info in
// // a ShadeableIntersection, as well as how to use thrust's random number
// // generator. Observe that since the thrust random number generator basically
// // adds "noise" to the iteration, the image should start off noisy and get
// // cleaner as more iterations are computed.
// //
// // Note that this shader does NOT do a BSDF evaluation!
// // Your shaders should handle that - this can allow techniques such as
// // bump mapping.
// __global__ void shadeFakeMaterial (int iter, int num_paths, ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < num_paths)
//   {
//     ShadeableIntersection intersection = shadeableIntersections[idx];
//     if (intersection.t > 0.0f) { // if the intersection exists...
//       // Set up the RNG
//       // LOOK: this is how you use thrust's RNG! Please look at
//       // makeSeededRandomEngine as well.
//       thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
//       // thrust::uniform_real_distribution<float> u01(0, 1);

//       Material material = materials[intersection.materialId];
//       glm::vec3 materialColor = material.color;

//       // If the material indicates that the object was a light, "light" the ray
//       if (material.emittance > 0.0f) {
//         pathSegments[idx].color *= (materialColor * material.emittance);
//         pathSegments[idx].remainingBounces = 0;
//       } 
//       else if (pathSegments[idx].remainingBounces == 1) {
//           pathSegments[idx].remainingBounces -= 1;
//           pathSegments[idx].color = glm::vec3(0.0f);
//       }
//       // Otherwise, do some pseudo-lighting computation. This is actually more
//       // like what you would expect from shading in a rasterizer like OpenGL.
//       // TODO: replace this! you should be able to start with basically a one-liner
//       else {
//         scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t, intersection.surfaceNormal,
//               material, rng);
//         // float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
//         // pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
//         // pathSegments[idx].color *= u01(rng); // apply some noise because why not
//       }
//     // If there was no intersection, color the ray black.
//     // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
//     // used for opacity, in which case they can indicate "no opacity".
//     // This can be useful for post-processing and image compositing.
//     } else {
//       pathSegments[idx].color = glm::vec3(0.0f);
//       pathSegments[idx].remainingBounces = 0;
//     }
//   }
// }

// // Add the current iteration's output to the overall image
// __global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
// {
// 	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

// 	if (index < nPaths)
// 	{
// 		PathSegment iterationPath = iterationPaths[index];
// 		image[iterationPath.pixelIndex] += iterationPath.color;
// 	}
// }

// struct end_condition {
//     __host__ __device__
//         bool operator()(const PathSegment& pathSegment) {
//         return (pathSegment.remainingBounces >= 0);
//     }
// };
// struct compare_materials {
//     __host__ __device__
//         bool operator()(const ShadeableIntersection& m1, const ShadeableIntersection& m2) {
//         return (m1.materialId > m2.materialId);
//     }
// };

// __global__ void shadeDirectLighting(int iter, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Material* materials, Geom* lights, int num
// ){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < num_paths)
//     {
//         if (pathSegments[idx].remainingBounces <= 0) {
//             return;
//         }

//         ShadeableIntersection intersection = shadeableIntersections[idx];
//         thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
//         PathSegment path = pathSegments[idx];

//         if (path.remainingBounces != 2 && path.remainingBounces > 0 && intersection.t > 0.f) {

//             thrust::uniform_real_distribution<float> u01(0, 1);

//             Material material = materials[intersection.materialId];
//             glm::vec3 materialColor = material.color;

//             // If the material indicates that the object was a light, "light" the ray
//             if (material.emittance > 0.0f) {
//                 pathSegments[idx].color *= (materialColor * material.emittance);
//                 pathSegments[idx].remainingBounces = 0;
//             }
//             else if (pathSegments[idx].remainingBounces == 1) {
//                 pathSegments[idx].remainingBounces -= 1;
//                 pathSegments[idx].color = glm::vec3(0.0f);
//             }
//             else {
//                 pathSegments[idx].remainingBounces -= 1;
//                 scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t, intersection.surfaceNormal,
//                     material, rng);
//             }
            
//         }
//         else if (path.remainingBounces == 2 && intersection.t > 0.f) {
//             Material material = materials[intersection.materialId];
//             glm::vec3 materialColor = material.color;

//             // If the material indicates that the object was a light, "light" the ray
//             if (material.emittance > 0.0f) {
//                 pathSegments[idx].color *= (materialColor * material.emittance);
//                 pathSegments[idx].remainingBounces = 0;
//             }
//             else {
//                 scatterRay(path, path.ray.origin + path.ray.direction * intersection.t, intersection.surfaceNormal, material, rng);
//                 thrust::uniform_real_distribution<float> u01(0, 1);
//                 float r = u01(rng);
//                 int lightIdx = 0;
//                 if (num != 0) {
//                     lightIdx = glm::min((int)glm::floor(r * num), num - 1);
//                 }
//                 glm::vec3 lightPt = pointOnPlane(lights[lightIdx], rng);
//                 path.ray.direction = glm::normalize(lightPt - path.ray.origin);
//                 path.remainingBounces--;
//             }
//         }
//         else {
//             pathSegments[idx].color = glm::vec3(0.0f);
//             pathSegments[idx].remainingBounces = 0;
//         }
//     }
// }
// /**
//  * Wrapper for the __global__ call that sets up the kernel calls and does a ton
//  * of memory management
//  */
// void pathtrace(uchar4 *pbo, int frame, int iter) {
//     const int traceDepth = hst_scene->state.traceDepth;
//     const Camera &cam = hst_scene->state.camera;
//     const int pixelcount = cam.resolution.x * cam.resolution.y;

// 	// 2D block for generating ray from camera
//     const dim3 blockSize2d(8, 8);
//     const dim3 blocksPerGrid2d(
//             (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
//             (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

//     // 1D block for path tracing
//     const int blockSize1d = 128;

//     ///////////////////////////////////////////////////////////////////////////

//     // Recap:
//     // * Initialize array of path rays (using rays that come out of the camera)
//     //   * You can pass the Camera object to that kernel.
//     //   * Each path ray must carry at minimum a (ray, color) pair,
//     //   * where color starts as the multiplicative identity, white = (1, 1, 1).
//     //   * This has already been done for you.
//     // * For each depth:
//     //   * Compute an intersection in the scene for each path ray.
//     //     A very naive version of this has been implemented for you, but feel
//     //     free to add more primitives and/or a better algorithm.
//     //     Currently, intersection distance is recorded as a parametric distance,
//     //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
//     //     * Color is attenuated (multiplied) by reflections off of any object
//     //   * TODO: Stream compact away all of the terminated paths.
//     //     You may use either your implementation or `thrust::remove_if` or its
//     //     cousins.
//     //     * Note that you can't really use a 2D kernel launch any more - switch
//     //       to 1D.
//     //   * TODO: Shade the rays that intersected something or didn't bottom out.
//     //     That is, color the ray by performing a color computation according
//     //     to the shader, then generate a new ray to continue the ray path.
//     //     We recommend just updating the ray's PathSegment in place.
//     //     Note that this step may come before or after stream compaction,
//     //     since some shaders you write may also cause a path to terminate.
//     // * Finally, add this iteration's results to the image. This has been done
//     //   for you.

//     // TODO: perform one iteration of path tracing



// 	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
// 	checkCUDAError("generate camera ray");

// 	int depth = 0;
// 	PathSegment* dev_path_end = dev_paths + pixelcount;
// 	int num_paths = dev_path_end - dev_paths;

// 	// --- PathSegment Tracing Stage ---
// 	// Shoot ray into scene, bounce between objects, push shading chunks

//   bool iterationComplete = false;
// 	while (!iterationComplete) {

//     // clean shading chunks
//     cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
//     // tracing
//     dim3 numBlocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    
//     // use cached first intersection
//     if (CACHE_BOUNCE && !ANTI_ALIASING && depth == 0 && iter != 1) {
//         thrust::copy(thrust::device, dev_intersection_first_bounce, dev_intersection_first_bounce + num_paths, dev_intersections);

//         //sort by material
//         if (SORT_MATERIALS) {
//             thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_materials());
//         }
//     }else {
//         // clean shading chunks
//         cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
//         computeIntersections <<<numBlocksPathSegmentTracing, blockSize1d>>> (depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
//         checkCUDAError("trace one bounce");
//         cudaDeviceSynchronize();

//         //cache first bounce
//         if (CACHE_BOUNCE && !ANTI_ALIASING && depth == 0 && iter == 1) {
//             thrust::copy(thrust::device, dev_intersections, dev_intersections + num_paths, dev_intersection_first_bounce);
//         }
//         //sort by material
//         if (SORT_MATERIALS) {
//             thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_materials());
//         }
//     }
//     depth++;

// #if DIRECT_LIGHTING
//     shadeDirectLighting<<<numBlocksPathSegmentTracing, blockSize1d>>>(iter, num_paths, dev_intersections, dev_paths, 
//                                                                             dev_materials, dev_lights, hst_scene->lights.size());
// #else
//     shadeFakeMaterial<<<numBlocksPathSegmentTracing, blockSize1d>>> (iter, num_paths, dev_intersections, dev_paths, dev_materials);
// #endif
//     dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, end_condition());
//     num_paths = dev_path_end - dev_paths;
//     if (num_paths == 0 || depth > traceDepth) {
//         iterationComplete = true;
//     }
// 	// TODO:
// 	// --- Shading Stage ---
// 	// Shade path segments based on intersections and generate new rays by
//   // evaluating the BSDF.
//   // Start off with just a big kernel that handles all the different
//   // materials you have in the scenefile.
//   // TODO: compare between directly shading the path segments and shading
//   // path segments that have been reshuffled to be contiguous in memory.
// // shade for direct lighting

//   // iterationComplete = true; // TODO: should be based off stream compaction results.
// 	}

//   // Assemble this iteration and apply it to the image
//   dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
// 	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

//     ///////////////////////////////////////////////////////////////////////////

//     // Send results to OpenGL buffer for rendering
//     sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

//     // Retrieve image from GPU
//     cudaMemcpy(hst_scene->state.image.data(), dev_image,
//             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

//     checkCUDAError("pathtrace");
// }



#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define ANTI_ALIASING 1
#define CACHE_BOUNCE 0
#define SORT_MATERIALS 0
#define DEPTH_OF_FIELD 0
#define DIRECT_LIGHTING 0

#define LENS_RADIUS 0.07
#define FOCAL_DISTANCE 5

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

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection* dev_first_bounce = NULL;
#if DIRECT_LIGHTING
static Geom* dev_lights = NULL;
#endif

void pathtraceInit(Scene *scene) {
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
#if CACHE_BOUNCE || SORT_MATERIALS
    cudaMalloc(&dev_first_bounce, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if DIRECT_LIGHTING
    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
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
#if CACHE_BOUNCE || SORT_MATERIALS
    cudaFree(dev_first_bounce);
#endif

#if DIRECT_LIGHTING
    cudaFree(dev_lights);
#endif
    checkCUDAError("pathtraceFree");
}

__host__ __device__
glm::vec3 pointOnPlane(Geom light, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 pt(u01(rng), u01(rng));
    glm::vec3 planePt = glm::vec3((pt - glm::vec2(0.5f)), 0.f);
    return glm::vec3(light.transform * glm::vec4(planePt, 1.f));
}


__host__ __device__
glm::vec3 convertDisk(const glm::vec2 &v) {
    float x = v.x;
    float y = v.y;
    float phi, r;
    float a = 2 * x - 1.f;
    float b = 2 * y - 1.f;

    if (a > -b) {
        if (a > b) {
            r = a;
            phi = (PI / 4) * (b / a);
        }
        else {
            r = b;
            phi = (PI / 4) * (2 - (a / b));
        }
    }
    else {
        if (a < b) {
            r = -a;
            phi = (PI / 4) * (4 + (b / a));
        }
        else {
            r = -b;
            if (b < 0 || b > 0) {
                phi = (PI / 4) * (6 - (a / b));
            }
            else {
                phi = 0;
            }
        }
    }
    return glm::vec3(cosf(phi) * r, sinf(phi) * r, 0);
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        //antialiasing by jittering the ray
        float x_aa = x;
        float y_aa = y;
        thrust::default_random_engine random = makeSeededRandomEngine(iter, index, traceDepth);
        

#if ANTI_ALIASING
        thrust::uniform_real_distribution<float> u01(-1.0f, 1.0f);
        x_aa += u01(random);
        y_aa += u01(random);
#endif
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x_aa - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y_aa - (float)cam.resolution.y * 0.5f)
            );

        //for depth of field
#if DEPTH_OF_FIELD
        thrust::uniform_real_distribution<float> u02(0, 1);
        glm::vec3 sample = convertDisk(glm::vec2(u02(random), u02(random)));
        glm::vec3 lens = (float)LENS_RADIUS * sample;
        glm::vec3 pt = segment.ray.origin + lens;
        glm::vec3 fp = segment.ray.origin + (float)FOCAL_DISTANCE * segment.ray.direction;

        segment.ray.origin = pt;
        segment.ray.direction = glm::normalize(fp - pt);

#endif
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size
    , ShadeableIntersection * intersections
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
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE) {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

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

// shade for direct lighting
__global__ void shadeDirectLighting(
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , Geom* lights
    , int num
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (pathSegments[idx].remainingBounces <= 0) {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
        PathSegment path = pathSegments[idx];

        if (path.remainingBounces != 2 && path.remainingBounces > 0 && intersection.t > 0.f) {

            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else if (pathSegments[idx].remainingBounces == 1) {
                pathSegments[idx].remainingBounces -= 1;
                pathSegments[idx].color = glm::vec3(0.0f);
            }
            else {
                pathSegments[idx].remainingBounces -= 1;
                scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t, intersection.surfaceNormal,
                    material, rng);
            }
            
        }
        else if (path.remainingBounces == 2 && intersection.t > 0.f) {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                scatterRay(path, path.ray.origin + path.ray.direction * intersection.t, intersection.surfaceNormal, material, rng);
                thrust::uniform_real_distribution<float> u01(0, 1);
                float r = u01(rng);
                int lightIdx = 0;
                if (num != 0) {
                    lightIdx = glm::min((int)glm::floor(r * num), num - 1);
                }
                glm::vec3 lightPt = pointOnPlane(lights[lightIdx], rng);
                path.ray.direction = glm::normalize(lightPt - path.ray.origin);
                path.remainingBounces--;
            }
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}


__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
      if (pathSegments[idx].remainingBounces <= 0) {
          return;
      }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0;
      }
      else if (pathSegments[idx].remainingBounces == 1) {
          pathSegments[idx].remainingBounces -= 1;
          pathSegments[idx].color = glm::vec3(0.0f);
      }
      else {
          pathSegments[idx].remainingBounces -= 1;
          scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t, intersection.surfaceNormal,
              material, rng);
      }
    // If there was no intersection, color the ray black.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
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

struct should_end {
    __host__ __device__
        bool operator()(const PathSegment& pathSegment) {
        return (pathSegment.remainingBounces >= 0);
    }
};

struct compare_materials {
    __host__ __device__
        bool operator()(const ShadeableIntersection& m1, const ShadeableIntersection& m2) {
        return (m1.materialId > m2.materialId);
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
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

    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
    while (!iterationComplete) {

    // tracing
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    //cache first bounce
    if (CACHE_BOUNCE && !ANTI_ALIASING && depth == 0 && iter != 1) {
        thrust::copy(thrust::device, dev_first_bounce, dev_first_bounce + num_paths, dev_intersections);

        //sort by material
        if (SORT_MATERIALS) {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_materials());
        }
    }
    else {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        //cache first bounce
        if (CACHE_BOUNCE && !ANTI_ALIASING && depth == 0 && iter == 1) {
            thrust::copy(thrust::device, dev_intersections, dev_intersections + num_paths, dev_first_bounce);
        }
        //sort by material
        if (SORT_MATERIALS) {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_materials());
        }
    }

    depth++;

#if DIRECT_LIGHTING
    shadeDirectLighting<<<numblocksPathSegmentTracing, blockSize1d>>>(iter, num_paths, dev_intersections, dev_paths, 
                                                                            dev_materials, dev_lights, hst_scene->lights.size());
#else
  shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (iter, num_paths, dev_intersections, dev_paths, dev_materials);
#endif

  dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, should_end());
  num_paths = dev_path_end - dev_paths;


    if (num_paths == 0 || depth > traceDepth) {
        iterationComplete = true;
    }
        
}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}