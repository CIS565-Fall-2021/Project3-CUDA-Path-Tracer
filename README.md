CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Lindsay Smith
*  [LinkedIn](https://www.linkedin.com/in/lindsay-j-smith/), [personal website](https://lindsays-portfolio-d6aa5d.webflow.io/).
* Tested on: Windows 10, i7-11800H 144Hz 16GB RAM, GeForce RTX 3060 512GB SSD (Personal Laptop)

![](img/dof_none.png)

## Features
* Basic pathtracer
* Refractive, Reflective, and Diffused surfaces
* Anti-Aliasing
* OBJ Loading (work in progress)
* Depth of Field (work in progress)

This pathtracer utilized [Physically Based Rendering](https://pbr-book.org/) for reference

### Materials
Diffused: The diffused material randomly selects a direction for the ray to bounce.

![](img/diffuse.png)

Specular (Reflective): The specular material reflects the ray across the normal, so there is a set direction that the ray will bounce in every time.

![](img/cornell.2021-10-07_10-31-08z.3067samp.png)

Refractive: The refractive material was implemented using Schlick's approximation for Fresnel effects. 

![](img/clear_glass.png)
![](img/blue_glass.png)

### Anti-Aliasing
In the first photo we can see the step-like edge of the sphere, but in the second one where anti-aliasing has been implemented it appears more smooth. This was implemented by slightly jittering the ray origin when calculating the direction. This provides the slight blur that we see around the edges of the sphere.

![](img/no_anti_aliasing.png) ![](img/better_anti_aliasing.png)

### OBJ Loading
My OBJ loader is still a work in progress, but I was able to load a cube obj and have it appear in this simple example as an emmissive surface (light source). When dealing
with more complicated surfaces there are issues with the way the shapes appear. I believe this is due to the way the triangles are being loaded, but I am not entirely sure how to fix it yet. I used [tinyObj](https://github.com/tinyobjloader/tinyobjloader) to implement this.

![](img/sphere.2021-10-10_03-14-14z.57samp.png)

### Depth of Field
For the depth of field we want to be able to shift the focus of our camera so that only certain parts of the image appear in focus. My implementation for this is not without flaws, but we can see how some areas are blurred. There is a slight distortion of the shapes that I am still working on fixing.

![](img/dof_1.png)
![](img/cornell.2021-10-07_14-47-26z.2215samp.png)

A bit too far:

![](img/cornell.2021-10-07_14-28-06z.344samp.png)
