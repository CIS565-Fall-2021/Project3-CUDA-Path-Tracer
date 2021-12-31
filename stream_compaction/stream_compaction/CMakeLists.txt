set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    "radix_sort.h"
    "utf-8.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "radix_sort.cu"
    "utf-8.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_library(stream_compaction ${sources} ${headers})
