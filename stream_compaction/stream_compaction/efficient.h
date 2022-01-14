#pragma once

#include "common.h"
#include "cVec.h"
#include "utf-8.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

	/* in-place scan over device array, doesn't start GPU Timer and assumes input is power of 2
         * the zero value of T is assumed to be (T) 0. (necessary for work-efficient scan)
         * templated only for int and size_t. For other overloads, efficient.cu must be modified */
	template <typename T> void scan_dev(int N, cu::cVec<T>* dev_data);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
