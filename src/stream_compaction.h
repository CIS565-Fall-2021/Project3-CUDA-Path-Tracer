#ifndef __STREAM_COMPACTION_THRUST_H__
#define __STREAM_COMPACTION_THRUST_H__

#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include "scene.h"

namespace stream_compaction {

/**
 * @brief Performs stream compaction on rays in-place using Thrust
 * (thrust::partition)
 *
 * @param dev_paths: input array of rays
 * @param num_ipaths: size of input array
 * @return int: size of output array
 */
int rayCompaction(PathSegment* dev_paths, const int num_paths);

}  // namespace stream_compaction

#endif /* __STREAM_COMPACTION_THRUST_H__ */
