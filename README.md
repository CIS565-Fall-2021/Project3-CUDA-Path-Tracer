Project 3 CUSA Path Tracer
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3*

* Raymond Yang
	* [LinkedIn](https://www.linkedin.com/in/raymond-yang-b85b19168)
	* Tested on: 
		* 10/09/2021
		* Windows 10
		* NVIDIA GeForce GTX 1080 Ti. 
	* Submitted on: 10/09/2021

## Introduction 
The objective of this project was to implement a naive core path tracer that took a simplistic approach to rendering scenes. 
<p align="center">
  <img src="img/a.png" alt="drawing" width="500" />
</p>
From a camera, a viewing plane (or image plane) can projected. We simulate the physical characters of light by shooting a ray (photon) from each pixel within our viewing plane towards the scene. The rays would iteratively be bounced from an origin point and a surface. In each iteration, the ray can either miss the scene entirely (entering a void) or can be obstructed by an entity within the scene. The ray can be obstructed either by a light source or non-light source. 
<p align="center">
  <img src="img/b.png" alt="drawing" width="500" />
</p>
For each iteration, if the ray is not obstructed or is obstructed by a light source, the ray is terminated. If the ray is obstructed by a non-light source, it will reflect, refract, and/or diffuse against the obstructing surface. 
<p align="center">
  <img src="img/c.png" alt="drawing" width="500" />
</p>
For each successful obstruction, the color of the obstructing surface is factored into the final color the ray's original corresponding pixel. 


## Core Features 
The [core features](https://github.com/CIS565-Fall-2021/Project3-CUDA-Path-Tracer/blob/main/INSTRUCTION.md#part-1---core-features) include:
* Naive BSDF Path Tracer (Feature Implementation)
* First Iteration Caching (Performance Improvement)
* Ray Stream Compaction (Performance Improvement) 
* Material Sorting (Performance Improvement) 
All features and performance improvements may be toggled by `#define`s found in `src/sceneStructs/h`

### Naive BSDF Path Tracer

### First Iteration Caching

### Ray Stream Compaction 

### Material Sorting

## Additional Features
The [unique features](https://github.com/CIS565-Fall-2021/Project3-CUDA-Path-Tracer/blob/main/INSTRUCTION.md#part-2---make-your-pathtracer-unique) include: 
* Mesh Loading using tinyOBJ (Feature Implementation)
	* Bounding Box (Performance Improvement)
* [Anti-Aliasing](https://raytracing.github.io/books/RayTracingInOneWeekend.html#antialiasing) (Feature Implementation)
* [Refraction using Schlick's Approximation](https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics) (Feature Implementation)

### Mesh Loading using tinyOBJ
This feature allows you to import unique .OBJ mesh into the path tracer. 

#### Bounding Box
Each mesh is a complex arrangement of numerous triangular faces with unique vertices and normals. The naive implementation would check every ray projected into the scene against every triangular surface of every mesh. This is clearly computationally expensive and time consuming. The first step to optimize this would be to restrict the volume of each mesh into a bounding box. That is, a mesh will only be checked against a ray for intersection if the ray will enter the bounding box of the mesh. The current implementation is minimally effective in that it is a single volume bounding box around the entire mesh. 

### Anti-Aliasing 
Anti-aliasing is a common feature that slightly distorts how a scene is rendered. This prevents far objects from being rendered with sharp edges that would typically result in texture jittering and collisions. The current implementation deviates the origin ray direction that is first projected from the camera into the scene on a random distribution. More precisely, the first ray of each iteration is shot out from a random position within the same pixel. That way, we obtain a better average of the color of the pixel. 
<p align="center">
  <img src="img/d.png" alt="https://raytracing.github.io/images/fig-1.07-pixel-samples.jpg" width="500" />
</p>

### Refraction using Schlick's Approximation 
If we looked at a refractive material surface such as a plane of glass or clear plastic from a steep angle, the material ceases to demonstrate refractive properties and would show reflective properties instead. The current implementation mimics this behavior using Schlick's approximation in cases where the incident angle between the surface and the ray is sufficiently shallow, and snell's law in cases where the incident angle between the surface and the ray is sufficiently large. 