set(headers
    "logCore.hpp"
    )

set(sources
    "logCore.cpp"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_library(profile_log ${sources} ${headers})
