#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

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

/*
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}
*/

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
static PathSegment * dev_paths2 = NULL;
static PathSegment * dev_pathBounce1Cache = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_intersections2 = NULL;
static ShadeableIntersection * dev_itxnBounce1Cache = NULL;
static Tri * dev_allTris = NULL;
static int* dev_meshStartIndices = NULL;
static int* dev_meshEndIndices = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_pathBounce1Cache, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_itxnBounce1Cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // --- mesh loading ---
    // load the triangles of all meshes into one big array
    int totalTris = 0;
    int numMeshes = 0;
    for (auto& g : hst_scene->geoms) {
        if (g.type == MESH) {
            totalTris += g.numTris;
            numMeshes++;
        }
    }

    // if there are any meshes in the 
    if (numMeshes) {
        cudaMalloc(&dev_allTris, totalTris * sizeof(Tri));
        cudaMalloc(&dev_meshStartIndices, numMeshes * sizeof(int));
        cudaMalloc(&dev_meshEndIndices, numMeshes * sizeof(int));

        // add the tris from all our geo
        int startIndex = 0;
        int meshNum = 0;
		for (auto& g : hst_scene->geoms) {
			if (g.type == MESH) {

                // copy the tris from this geo, offset the
                // start index for the next copy
				cudaMemcpy(dev_allTris + startIndex, 
						   g.tris + startIndex, 
						   g.numTris * sizeof(Tri), 
						   cudaMemcpyHostToDevice);

                // copy the start index for this mesh
                cudaMemcpy(dev_meshStartIndices + meshNum,
						   &startIndex,
						   sizeof(int),
						   cudaMemcpyHostToDevice);

                // incr the start index for the next mesh
				startIndex += g.numTris;

                // start index for the next mesh is the end index for
                // this mesh
                cudaMemcpy(dev_meshEndIndices + meshNum,
						   &startIndex,
						   sizeof(int),
						   cudaMemcpyHostToDevice);
			}
    }

    }
    else {
        // declare an empty (nearly) array just because it needs to exist
        // for freeing/reference etc. 
        cudaMalloc(&dev_allTris, sizeof(Tri));
        cudaMalloc(&dev_meshStartIndices, numMeshes * sizeof(int));
        cudaMalloc(&dev_meshEndIndices, numMeshes * sizeof(int));
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_paths2);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_intersections2);
    cudaFree(dev_itxnBounce1Cache);
    cudaFree(dev_pathBounce1Cache);
    cudaFree(dev_allTris);
    cudaFree(dev_meshStartIndices);
    cudaFree(dev_meshEndIndices);

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

// Based on Pharr/Humphrey's Physically Based Rendering textbook, 2nd ed
// Sections 6.2.3 and 13.6
__device__ void concentricSampleDisk(thrust::default_random_engine& rng,
								  float* dx,
								  float* dy,
                                  float apSize) {
    thrust::uniform_real_distribution<float> un11(-1.0f * apSize, apSize);
    float r;
    float theta;
    float sy = un11(rng);
    float sx = un11(rng);

    if (sx == 0 && sy == 0) {
        *dx = 0;
        *dy = 0;
        return;
    }
    if (abs(sx) > abs(sy)) {
        r = sx;
        theta = (PI * sx) / (sx * 4.0f);
    }
    else {
        r = sy;
        theta = (PI / 2.0f) - ((PI * sx) / (sy * 4.0f));
    }
    float u1 = un11(rng);
    float u2 = un11(rng);
    r = sqrt(u1);
    theta = 2.0f * PI * u2;

    *dx = r * cos(theta);
    *dy = r * sin(theta);
}

// Based on Pharr/Humphrey's Physically Based Rendering textbook, 2nd ed
// Sections 6.2.3 and 13.6
__device__ void samplePointOnLens(thrust::default_random_engine rng,
                                  float *lensU,
                                  float *lensV,
                                  float lensRadius,
                                  float apSize) {
    concentricSampleDisk(rng, lensU, lensV, apSize);
    *lensU *= lensRadius;
    *lensV *= lensRadius;
}

__global__ void generateRayFromCamera(Camera cam, 
    int iter, 
    int traceDepth, 
    PathSegment* pathSegments, 
    bool useDOF,
    bool antialias)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];
        segment.ray.origin = cam.position;

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);

        float aaShiftx = 0.0f;
        float aaShifty = 0.0f;
        if (antialias) {
			thrust::uniform_real_distribution<float> u01(-1.0f, 1.0f);
            aaShiftx = u01(rng);
            aaShifty = u01(rng);
        }
            
        // calculate initial rays based on pin-hole camera
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + aaShiftx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + aaShifty - (float)cam.resolution.y * 0.5f)
            );

        if (useDOF) {
            // find the point on plane of focus, i.e. the plane on which all rays bent 
            // by the lens well converge
            glm::vec3 pfocus = cam.position + segment.ray.direction * cam.focalDist;

            // Offset the ray origins. Rather than all being from one point, they are now
            // effectively cast from an aperture
            float u, v;
            samplePointOnLens(rng, &u, &v, cam.lensRadius, cam.aperture);
            segment.ray.origin = cam.position + u * cam.right + v * cam.up;

            // recalculate ray direction based on aperture/lens model. Ray now
            // points to the point of focus
            segment.ray.direction = glm::normalize(pfocus - segment.ray.origin);
        }

        // initialixe other aspects of path segment
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

//__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
//{
//    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//    if (x < cam.resolution.x && y < cam.resolution.y) {
//        int index = x + (y * cam.resolution.x);
//        PathSegment & segment = pathSegments[index];
//
//        segment.ray.origin = cam.position;
//        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
//
//
//        segment.ray.direction = glm::normalize(cam.view
//            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
//            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
//            );
//
//        segment.pixelIndex = index;
//        segment.remainingBounces = traceDepth;
//    }
//}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in the shader(s).
__global__ void computeIntersections(int depth, 
									 int num_paths, 
									 PathSegment * pathSegments, 
									 Geom * geoms, 
									 int geoms_size, 
									 ShadeableIntersection * intersections,
                                     Tri * dev_allTris,
                                     int * dev_meshStartIndices,
                                     int * dev_meshEndIndices)
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

        int meshNum = 0;
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
            else if (geom.type == MESH) {
                int numTris = dev_meshEndIndices[meshNum] - dev_meshStartIndices[meshNum];

                meshIntersectionTest(geom, 
									 pathSegment.ray, 
									 tmp_intersect, 
									 tmp_normal, 
									 outside,
									 dev_allTris + dev_meshStartIndices[meshNum],
									 numTris);

                meshNum++;
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
			pathSegments[path_index].color = glm::vec3(0);
            pathSegments[path_index].remainingBounces = 0;
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


// allShader has conditionals for all BSDFs. It's inefficient, but it gets us stared 
__global__ void shadeAllMaterial (
    int iter,
    int num_paths,
    ShadeableIntersection * shadeableIntersections,
    PathSegment * pathSegments,
    Material * materials){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
      thrust::uniform_real_distribution<float> u01(0, 1);

      PathSegment path = pathSegments[idx];

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0){// && pathSegments[idx].remainingBounces > 0) {
	    path.color *= (materialColor * material.emittance);
        path.remainingBounces = 0;
      }

      else{// if (pathSegments[idx].remainingBounces > 0){
	    path.color *= materialColor;
        scatterRay(path,
				  getPointOnRay(path.ray, intersection.t),
				  intersection.surfaceNormal,
				  material,
				  rng);
        path.remainingBounces--;
      }
      pathSegments[idx] = path;
  }
}

__device__ glm::vec3 devClampRGB(glm::vec3 col) {
    glm::vec3 out;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        out[i] = min(max(0.0f,col[i]), 255.0f);
    }
    return out;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths, float iterations){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        // yes we have to clamp here even though there is later clamping
        // otherwise reflective surfaces generate fireflies
		image[iterationPath.pixelIndex] += devClampRGB(iterationPath.color);// ((image[iterationPath.pixelIndex] * (iterations - 1)) + iterationPath.color) / iterations;// *0.001f;
    }
}

// predicate for culling paths based on bounce depth
struct hasBounces{
  __device__ bool operator()(const PathSegment &path){
    return (path.remainingBounces > 0);
  }
};

__global__ void kernScatterPathsAndIntersections(int n,
                                                 PathSegment *paths,
                                                 const PathSegment *pathsRO,
                                                 ShadeableIntersection *intersections,
                                                 const ShadeableIntersection *intersectionsRO,
                                                 const int *indices) {
       int index = (blockIdx.x * blockDim.x) + threadIdx.x;
       if (index >= n) {
               return;
       }

       paths[index] = pathsRO[indices[index]];
       intersections[index] = intersectionsRO[indices[index]];
}

__global__ void kernEnumerate(int n, int* indices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n) {
        indices[index] = index;
    }
}

__global__ void kernGetMaterialIds(int n, 
                                    int* materialIds,
                                    int* indices,
                                    const ShadeableIntersection* dev_intersections) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n) {
        materialIds[index] = dev_intersections[indices[index]].materialId;
    }
}

int cullPathsAndSortByMaterial(int num_paths,
							   const PathSegment* dev_pathsRO,
							   PathSegment* dev_pathsOut,
							   const ShadeableIntersection* dev_intersectionsRO,
							   ShadeableIntersection* dev_intersectionsOut,
							   const int blockSize1d) {
    // cull and sort in one kernel to save on global reads/writes when 
    // rearranging paths/intersections

    // --- Cull ---
    int newNumPaths;
    int* indices;
    // the addr of the last non-culled path
    int* partitionMiddle;
    
	cudaMalloc((void**)&indices, num_paths * sizeof(int));

    int numBlocks = ceil((float)num_paths / blockSize1d);
    kernEnumerate <<<numBlocks, blockSize1d >>> (num_paths, indices);


    // effectively sort indices based on whether an object was hit
    partitionMiddle = thrust::stable_partition(thrust::device, indices, indices + num_paths, dev_pathsRO, hasBounces());
    // do some pointer math to return the index
    newNumPaths = partitionMiddle - indices;
    
    // --- Sort by Material ---
    // now everything before noHitIndex has hit something. Sort them by their material
    if (hst_scene->state.sortMaterials) {
        int* materialIds;
        cudaMalloc((void**)&materialIds, newNumPaths * sizeof(int));

        // get material ids. We need to pass indices since we haven't reshuffled intersections yet
        numBlocks = ceil((float)newNumPaths / blockSize1d);
        kernGetMaterialIds << <numBlocks, blockSize1d >> > (newNumPaths, materialIds, indices, dev_intersectionsRO);

        thrust::sort_by_key(thrust::device, materialIds, materialIds + newNumPaths, indices);

        cudaFree(materialIds);
    }

	// assign paths/intersections to the sorted indices. now all paths/intersections before `newNumPaths` have hit an obj
    // note we have to assign ALL paths and intersections (i.e. use `num_paths` not `newNumPaths`) because some paths wouldn't
    // be assigned and/or would be overwritten
	numBlocks = ceil((float)num_paths / blockSize1d);
	kernScatterPathsAndIntersections << <numBlocks, blockSize1d >> > (num_paths,
																	  dev_pathsOut,
																	  dev_pathsRO,
																	  dev_intersectionsOut,
																	  dev_intersectionsRO,
																	  indices);
	//checkCUDAError("scatter");


    cudaFree(indices);

    return newNumPaths;
}

/*
emitter
lambert type
phong type
reflective phong
reflective/refractive phong
*/


void shade(int iter,
			int num_paths,
			ShadeableIntersection* dev_intersections,
			PathSegment* dev_paths,
			Material* dev_materials,
			dim3 numblocksPathSegmentTracing,
			int blockSize1d) {
    shadeAllMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (iter,
																	num_paths,
																	dev_intersections,
																	dev_paths,
																	dev_materials);

    checkCUDAError("shade");
}

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
    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int preCulledNumPaths = num_paths;
    PathSegment* pathSwp;
    ShadeableIntersection* intxnSwp; // for ping ponging buffers
    if (hst_scene->state.cacheFirstBounce && iter > 1) {
        cudaMemcpy(dev_intersections, 
				   dev_itxnBounce1Cache, 
				   num_paths * sizeof(ShadeableIntersection), 
				   cudaMemcpyDeviceToDevice);
        checkCUDAError("copying itxn cache");
        cudaMemcpy(dev_paths, 
				   dev_pathBounce1Cache, 
				   num_paths * sizeof(PathSegment), 
				   cudaMemcpyDeviceToDevice);
        checkCUDAError("copying path cache");
        depth=1;
    }
    else {
        // cast camera rays using either the DOF kernel or the pinhole kernel
		generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, 
																iter, 
																traceDepth, 
																dev_paths,
																hst_scene->state.useDOF,
                                                                hst_scene->state.antialias);
        checkCUDAError("generate camera ray");
    }


    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    

    bool iterationComplete = false;

    while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (depth,
																			num_paths,
																			dev_paths,
																			dev_geoms,
																			hst_scene->geoms.size(),
																			dev_intersections,
                                                                            dev_allTris,
                                                                            dev_meshStartIndices,
                                                                            dev_meshEndIndices);
		depth++;


		// --- cull dead-end paths ---
        num_paths = cullPathsAndSortByMaterial(preCulledNumPaths, 
											   dev_paths, 
											   dev_paths2, 
										       dev_intersections, 
											   dev_intersections2, 
											   blockSize1d);
		checkCUDAError("cull");
        // ping-pong buffers after culling.
        pathSwp = dev_paths;
        dev_paths = dev_paths2;
        dev_paths2 = pathSwp;
        intxnSwp = dev_intersections;
        dev_intersections = dev_intersections2;
        dev_intersections2 = intxnSwp;

        if (iter == 1 && depth == 1 && hst_scene->state.cacheFirstBounce) {
			cudaMemcpy(dev_itxnBounce1Cache, 
					   dev_intersections, 
					   preCulledNumPaths * sizeof(ShadeableIntersection), 
					   cudaMemcpyDeviceToDevice);
        checkCUDAError("reading itxn cache");
			cudaMemcpy(dev_pathBounce1Cache, 
					   dev_paths, 
					   preCulledNumPaths * sizeof(PathSegment), 
					   cudaMemcpyDeviceToDevice);
        checkCUDAError("reading path cache");
        }
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		shade(iter,
						 num_paths,
						 dev_intersections,
						 dev_paths,
						 dev_materials,
						 numblocksPathSegmentTracing,
						 blockSize1d);


		if (depth >= traceDepth || num_paths == 0) {
			iterationComplete = true; // TODO: should also be based off stream compaction results.
		}
    }

    
    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(preCulledNumPaths, dev_image, dev_paths, iter);

    // reset num_paths. We've culled some but want the full number next iteration
    //num_paths = preCulledNumPaths;
    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
