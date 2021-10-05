#ifndef __PATH_TRACER_STATIC_CONFIG_H__
#define __PATH_TRACER_STATIC_CONFIG_H__

#include <cuda.h>
#include <device_launch_parameters.h>

#define CACHE_INTERSECTIONS
// #define DEPTH_OF_FIELD
#define EPS              0.0001f
#define ANTIALIAS_FACTOR 4

namespace static_config {

// Whether to use radix sort
extern const bool enable_radixSort;

}  // namespace static_config

#endif /* __PATH_TRACER_STATIC_CONFIG_H__ */
