CUDA Path Tracer
================

Implementation of a CUDA-based path tracer capable of rendering globally-illuminated images.

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3 - Path Tracer**

- Anthony Mansur
  - https://www.linkedin.com/in/anthony-mansur-ab3719125/
- Tested on: Windows 10, AMD Ryzen 5 3600, Geforce RTX 2060 Super (personal)



## Introduction

### What is a Path Tracer?

A Path Tracer is a rendering technique used to generate graphical images based on physically simulating how light works. As opposed to rasterization, it starts by shooting rays from a camera to the pixels in our screen, and following those rays at several points of intersection until it hits a light source. At every bounce, we multiply our current color with the color of the material hit. 

This technique has been widely used, mostly in the CPU, in many industries, most notably in animation, to generate photo-realistic images that mimic what we see in the real-world.



### What's different?

In this project, we attempt to create a path tracer utilizing the GPU, giving us much faster rendering speeds. As we start with a very basic implementation, there are many avenues of improvement in our path tracer, and this readme will go over some of those strategies used to make such improvements. We will first start of with the core features before diving deeper in how we made this path tracer unique! 

## Features

### Baseline

The following features were given to us before starting this assignment:

- Loading and reading the scene description format.
- Sphere and box intersection functions.
- Support for saving images.
- Working CUDA-GL interop for previewing your render while running.
- A skeleton render with:
  - Naive ray-scene intersection
  - A "fake" shading kernel that colors rays based on the material and intereseciton properties but does NOT compute a new ray based on the BSDF.



### Core Features

- Basic path-tracer with an improved shading kernel with BSDF evaluation for Ideal Diffuse Surfaces and Perfectly specular-reflective surfaces
  - As opposed to terminating after the first bounce, each ray generated will bounce to a new direction depending on the type of material hit
    - If a diffuse object is hit, the new direction bounced will be a cosine-weighted random direction in a hemisphere
    - If a reflective surface is hit, we combine the visual effects of a diffuse material and a perfectly specular surface that bounces the ray at a perfect reflection
      - 50% of the time, we bounce the ray using one method, but divide the overall color by it's probabiltiy 0.5.
  - A ray is terminated if it hits a light source, does not intersect an object, or runs out of bounces
  - Starting from RGB(1,1,1) , at each intersection, the ray's color is multiplied with the material color of it's intersection
- Utilizing thrust's library, at every bounce our rays take, in a parallel fashion, we remove all the rays that have "terminated," i.e., those rays thatno longer bounce, to minimize the amount of threads launched that did no work.
- Sort the rays based on the materials the intersected with
  - Issue: at every iteration, rays intersect different materials, and the BSDF evaluation may be different. So, if the rays (threads) at a warp have different BSDF evaluations, there will be branch divergence, which is not optimal as every thread needs to wait until the last one for all of them to be released.
  - Solution: We have three buffers of interest, our rays, intersections, and materials. The intersections are indexed the same way as our rays, and each interesection has an value to the index of the material of the object hit. We utilize thrust's sort_by_key operation, and sort the intersections and rays based on the material index. 
  - Performance: This sorting is costly, as it adds more computations and memory reallocation at every iteration. If your scene does not have that many different materials, you will most likely see a significant performance hit. However, if you have a lot of materials, this additional step may increase your performance (see the Issue section above).
- A toggleable option to cache the first bounce intersection for re-use across all subsequent iterations
  - See Performance analysis for results.



## Performance Analysis

TODO: Add analysis of caching the first bounce for different depths (from 1-20?)
