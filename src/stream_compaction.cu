#include "stream_compaction.h"

namespace stream_compaction {
struct is_ray_active_t {
  __host__ __device__ bool operator()(const PathSegment& path) {
    return (path.remainingBounces > 0);
  }
};

int rayCompaction(PathSegment* dev_paths, const int num_paths) {
  auto dv_end    = thrust::partition(thrust::device, dev_paths,
                                  dev_paths + num_paths, is_ray_active_t());
  int num_opaths = dv_end - dev_paths;
  return num_opaths;
}

}  // namespace stream_compaction
