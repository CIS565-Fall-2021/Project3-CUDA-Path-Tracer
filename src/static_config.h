#ifndef __PATH_TRACER_STATIC_CONFIG_H__
#define __PATH_TRACER_STATIC_CONFIG_H__

#include <cuda.h>
#include <device_launch_parameters.h>

#define EPS 0.0001f

namespace static_config {

// Whether to use radix sort
extern const bool enable_radixSort;

}  // namespace static_config

#endif /* __PATH_TRACER_STATIC_CONFIG_H__ */
