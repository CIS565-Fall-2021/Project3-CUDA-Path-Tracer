#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "cu.h"

using glm::ivec2;
using glm::ivec3;
using glm::vec3;
using cu::cPtr;
using cu::cVec;

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
	int h = hash((1 << 31) | (depth << 22) | iter) ^ hash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, ivec2 resolution, int iter, vec3 *image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		vec3 pix = image[index];

		ivec3 color;
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

static Scene *hst_scene = NULL;

static cPtr<vec3> dv_img = NULL;
static cPtr<Geom> dv_geoms = NULL;
static cPtr<Material> dv_materials = NULL;
static cPtr<PathSegment> dv_paths = NULL;
static cPtr<ShadeableIntersection> dv_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene)
{
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	dv_img = cu::make<vec3>(pixelcount);
	cu::set(dv_img, 0, pixelcount);

	dv_paths = cu::make<PathSegment>(pixelcount);

	dv_geoms = cu::make<Geom>(scene->geoms.size());
	cu::copy(dv_geoms, scene->geoms.data(), scene->geoms.size());

	dv_materials = cu::make<Material>(scene->materials.size());
	cu::copy(dv_materials, scene->materials.data(), scene->materials.size());

	dv_intersections = cu::make<ShadeableIntersection>(pixelcount);
	cu::set(dv_intersections, 0, pixelcount);

	// TODO: initialize any extra device memeory you need

}

void pathtraceFree()
{
	cu::del(dv_img);  // no-op if dv_img is null
	cu::del(dv_paths);
	cu::del(dv_geoms);
	cu::del(dv_materials);
	cu::del(dv_intersections);

	// TODO: clean up any extra device memory you created

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

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment &segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float) x - (float) cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float) y - (float) cam.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment *pathSegments,
	Geom *geoms,
	int geoms_size,
	ShadeableIntersection *intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		vec3 intersect_point;
		vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		vec3 tmp_intersect;
		vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++) {
			Geom &geom = geoms[i];

			if (geom.type == GeomType::CUBE) {
				t = boxIntersectionTest(geom, pathSegment.ray, &tmp_intersect, &tmp_normal, &outside);
			} else if (geom.type == GeomType::SPHERE) {
				t = sphereIntersectionTest(geom, pathSegment.ray, &tmp_intersect, &tmp_normal, &outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
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
	int iter,
	int num_paths,
	ShadeableIntersection *shadeableIntersections,
	PathSegment *pathSegments,
	Material *materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
		return;
	ShadeableIntersection intersection = shadeableIntersections[idx];
	if (intersection.t > 0.0f) { // if the intersection exists...
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		vec3 materialColor = material.color;

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.0f) {
			pathSegments[idx].color *= (materialColor * material.emittance);
		}
		// Otherwise, do some pseudo-lighting computation. This is actually more
		// like what you would expect from shading in a rasterizer like OpenGL.
		// TODO: replace this! you should be able to start with basically a one-liner
		else {
			float lightTerm = glm::dot(intersection.surfaceNormal, vec3(0.0f, 1.0f, 0.0f));
			pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
			pathSegments[idx].color *= u01(rng); // apply some noise because why not
		}
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
	} else {
		pathSegments[idx].color = vec3(0.0f);
	}
}

__global__ void shade_material(int iter, int num_paths, ShadeableIntersection *s_intersections,
	PathSegment *path_segments, Material *materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
		return;
	if (path_segments[idx].remainingBounces <= 0)
		return;

	ShadeableIntersection intersection = s_intersections[idx];
	if (intersection.t > 0.0f) {
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
//		thrust::uniform_real_distribution<float> u01(0, 1);

		vec3 intersect_point = getPointOnRay(path_segments[idx].ray, intersection.t);
		Material material = materials[intersection.materialId];

		scatterRay(path_segments[idx], intersect_point, intersection.surfaceNormal, material, rng);
	} else {
		path_segments[idx].color = vec3(0.0f);
		path_segments[idx].remainingBounces = 0;
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, vec3 *image, PathSegment *iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= nPaths)
		return;

	PathSegment iterationPath = iterationPaths[index];
	image[iterationPath.pixelIndex] += iterationPath.color;
}


// FIXME: replace this with my own impl for stream compact
struct is_zero {
	__host__ __device__ bool operator()(const PathSegment &s)
	{
		glm::length(s.color) < EPSILON;
	}
};


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
	//	 A very naive version of this has been implemented for you, but feel
	//	 free to add more primitives and/or a better algorithm.
	//	 Currently, intersection distance is recorded as a parametric distance,
	//	 t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//	 * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//	 You may use either your implementation or `thrust::remove_if` or its
	//	 cousins.
	//	 * Note that you can't really use a 2D kernel launch any more - switch
	//	   to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//	 That is, color the ray by performing a color computation according
	//	 to the shader, then generate a new ray to continue the ray path.
	//	 We recommend just updating the ray's PathSegment in place.
	//	 Note that this step may come before or after stream compaction,
	//	 since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d>>> (cam, iter, traceDepth, dv_paths.get());
	cu_check_err("generate camera ray");

	int depth = 0;
	cPtr<PathSegment> dv_path_end = dv_paths + pixelcount;
	size_t num_paths = dv_path_end - dv_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	do {
		// clean shading chunks
		cu::set(dv_intersections, 0, pixelcount);

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth,
			num_paths,
			dv_paths.get(),
			dv_geoms.get(),
			hst_scene->geoms.size(),
			dv_intersections.get());
		cu_check_err("trace one bounce");
		cudaDeviceSynchronize();
		depth++;


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		//shadeFakeMaterial <<<numblocksPathSegmentTracing, blockSize1d>>> (
		//	iter,
		//	num_paths,
		//	dv_intersections.get(),
		//	dv_paths.get(),
		//	dv_materials.get()
		//	);
		//cu_check_err("shadeFakeMaterial");

		shade_material <<<numblocksPathSegmentTracing, blockSize1d >>> (
			iter,
			num_paths,
			dv_intersections.get(),
			dv_paths.get(),
			dv_materials.get()
			);
		cu_check_err("shade_material");

		// FIXME: replace this with my own impl for stream compact
		thrust::device_ptr<PathSegment> start(dv_paths.get()), end(dv_path_end.get());
		dv_path_end = thrust::remove_if(start, end, is_zero()).get();

	} while (depth < traceDepth && dv_path_end > dv_paths); // DONE: should be based off stream compaction results.

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather <<<numBlocksPixels, blockSize1d>>> (num_paths, dv_img.get(), dv_paths.get());
	cu_check_err("finalGather");

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO <<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dv_img.get());
	cu_check_err("sendImageToPBO");

	// Retrieve image from GPU
	cu::copy(hst_scene->state.image.data(), dv_img.get(), pixelcount);
}


