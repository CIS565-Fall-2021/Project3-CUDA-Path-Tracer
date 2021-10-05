CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zirui Zang
  * [LinkedIn](https://www.linkedin.com/in/zirui-zang/)
* Tested on: Windows 10, AMD Ryzen 7 3700X @ 3.60GHz 32GB, RTX2070 SUPER 8GB (Personal)

### Photorealistic Rendering by Tracing Rays

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Ray_trace_diagram.png/320px-Ray_trace_diagram.png"
     alt="Pathtrace" />
</p>

### Surface Scattering 

Below list the 3 separate scattering surfaces implemented in this renderer. To make refracgive surface seen more realistic, schlicks approximation is used to make the surface the effect of Frenel reflections. Objects in the seen can use a combination of these 3 properties based on the bidirectional scattering distribution function (BSDF) of the surface. 

| Lambertian | Reflective | Refractive |
| ------------- | ----------- |----------- |
| ![](scenes/cornell.2021-10-03_23-27-37z.3488samp.png)  | ![](scenes/cornell.2021-10-03_23-34-37z.4309samp.png) | ![](scenes/cornell.2021-10-03_23-43-33z.1964samp.png) |

Here shows different surface with colors.

| scenes/cornell.2021-10-04_00-14-05z.1925samp.png |

<p align="center">
<img src="scenes/cornell.2021-10-04_00-14-05z.1925samp.png"
     alt="Pathtrace" />
</p>

### Surface Scattering 
