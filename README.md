CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zhihao Ruan (ruanzh@seas.upenn.edu)
  * [LinkedIn](https://www.linkedin.com/in/zhihao-ruan-29b29a13a/), [personal website](https://zhihaoruan.xyz/)
* Tested on: Ubuntu 20.04 LTS, Ryzen 3700X @ 2.22GHz 48GB, RTX 2060 Super @ 7976MB

![](img/cornell.2021-09-30_02-51-25z.5000samp.png)


## Highlights
Finished path tracing core features:
- diffuse shaders
- perfect specular reflection
- 1st-bounce ray intersection caching
- radix sort by material type
- path continuation/termination by Thrust stream compaction 

Finished Advanced Features:
- Refraction with Fresnel effects using Schlick's approximation
- Stochastic sampled anti-aliasing

### Ray Refraction for Glass-like Materials
|            Perfect Specular Reflection             |               Glass-like Refraction                |
| :------------------------------------------------: | :------------------------------------------------: |
| ![](img/cornell.2021-10-04_02-10-06z.5000samp.png) | ![](img/cornell.2021-10-04_01-57-31z.5000samp.png) |

### Stochastic Sampled Anti-Aliasing
|                 1x Anti-Aliasing (Feature OFF)                  |                        4x Anti-Aliasing                         |
| :-------------------------------------------------------------: | :-------------------------------------------------------------: |
| ![](img/cornell.2021-10-04_01-14-01z.5000samp-antialias-1x.png) | ![](img/cornell.2021-10-04_01-07-24z.5000samp-antialias-4x.png) |

