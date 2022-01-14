#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "common.h"

namespace cu {


template <typename T> class cPtr;
template <typename T> class cVec;

template <typename T>
/* allocates device array T[len] of length len */
cPtr<T> make(size_t len) {
		T *d;
		cudaMalloc((void**) &d, sizeof(*d) * len);
		checkCUDAError("cVec: cudaMalloc failed!");
		return cPtr<T>(d);
}

template <typename T>
/* deletes device array */
static void del(cPtr<T> p) {
	cudaFree(p.raw_ptr());
	checkCUDAError("cVec: cudaFree failed!");
}


/* a "raw" pointer for device memory, points to cuda allocated memory, not a smart pointer */
template <typename T>
class cPtr {
	T *p;
public:

	constexpr cPtr() : p(nullptr) {}
	cPtr(T *dev_ptr) : p(dev_ptr) {}
	cPtr(const cPtr<T> &cp) : p(cp.p) {}
	cPtr<T>& operator=(const cPtr<T>& cp) { p = cp.p; return *this; }
	cPtr(cPtr&& cp) : p(cp.p) { cp.p = nullptr; }
	cPtr& operator=(cPtr&& cp) { p = cp.p; cp.p = nullptr; return *this; }
	~cPtr() {}

	cPtr<T>& operator=(const cVec<T>& v) { p = v.p; return *this; }

	cPtr& operator++() { p++; return *this; }
	cPtr operator++(int) { cPtr<T> cp = *this; p++; return cp; }
	cPtr& operator--() { p--; return *this; }
	cPtr operator--(int) { cPtr<T> cp = *this; p--; return cp; }
	cPtr& operator+=(size_t i) { p += i; return *this; }
	cPtr& operator-=(size_t i) { p -= i; return *this; }
	cPtr operator+(size_t i) const { cPtr<T> cp = *this; cp.p += i; return cp; }
	cPtr operator-(size_t i) const { cPtr<T> cp = *this; cp.p -= i; return cp; }

	bool operator!() const { return p != nullptr; }

	bool operator==(const cPtr<T> &b) const { return p == b.p; }
	bool operator!=(const cPtr<T> &b) const { return p != b.p; }
	bool operator<=(const cPtr<T> &b) const { return p <= b.p; }
	bool operator>=(const cPtr<T> &b) const { return p >= b.p; }
	bool operator< (const cPtr<T> &b) const { return p < b.p; }
	bool operator> (const cPtr<T> &b) const { return p > b.p; }

	T *raw_ptr()  { return p; }
	const T *raw_ptr() const { return p; }
};

template <typename T>
/* set n elements to val */
void set(cPtr<T> dst, const T val, size_t n = 1) {
	cudaMemset(dst.raw_ptr(), val, n * sizeof(T));
	checkCUDAError("cuda memset failed!");
}

template <typename T>
/* copies n elements from device pointer src to device pointer dst */
void copy(cPtr<T> dst, const cPtr<T> src, size_t n) {
	cudaMemcpy(dst.raw_ptr(), src.raw_ptr(), n * sizeof(T), cudaMemcpyDeviceToDevice);
	checkCUDAError("cuda memcpy device to device failed!");
}

template <typename T>
/* copies n elements from host pointer src to device pointer dst */
void copy(cPtr<T> dst, const T* src, size_t n) {
	cudaMemcpy(dst.raw_ptr(), src, n * sizeof(T), cudaMemcpyHostToDevice);
	checkCUDAError("cuda memcpy host to device failed!");
}

template <typename T>
/* copies n elements from device pointer src to host pointer dst */
void copy(T *dst, const cPtr<T> src, size_t n) {
	cudaMemcpy(dst, src.raw_ptr(), n * sizeof(T), cudaMemcpyDeviceToHost);
	checkCUDAError("cuda memcpy device to host failed!");
}

template <typename T>
/* copies n elements from host pointer src to host pointer dst */
void copy(T* dst, const T* src, size_t n) {
	cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToHost);
	checkCUDAError("cuda memcpy host to host failed!");
}




/* smart container for manually allocated CUDA memory, holds both single vars and arrays */

template <typename T>
class cVec {
	size_t n;
	cPtr<T> p;

public:
	constexpr cVec() : n(0), p(nullptr) {}


	/* constructor, takes in host pointer, length of pointer array */
	cVec(size_t len, const T *host_ptr) : n(len) {
		p = make<T>(n);
		copy(p, host_ptr, n);
	}

	/* constructor, takes in host pointer, length of pointer array, and desired length of device array.  dev_len cannot be shorter than host_len */
	cVec(size_t host_len, const T *host_ptr, size_t dev_len) : n(dev_len) {
		p = make<T>(n);
		copy(p, host_ptr, host_len);
	}

	/* constructor, does not zero memory */
	cVec(size_t len) : n(len) {
		p = make<T>(n);
	}

	/* copy constructor */
	cVec(const cVec<T>& v) : n(v.n) {
		p = make<T>(n);
		copy(p, v.p, n);
	}

	/* copy assignment */
	cVec<T>& operator=(const cVec<T>& v) {
		if (v.n != n) {
			n = v.n;
			del(p);
			p = make<T>(n);
		}
		copy(p, v.p, n);
	}

	/* destructor */
	~cVec() {
		del(p);
		n = 0;
		p = nullptr;
	}

	/* move constructor */
	cVec(cVec&& v) : n(v.n), p(v.p) {
		v.n = 0;
		v.p = nullptr;
	}

	/* move assignment */
	cVec& operator=(cVec&& v){
		del(p);
		n = v.n;
		p = v.p;
		v.n = 0;
		v.p = nullptr;
		return *this;
	}

	size_t length() {
		return n;
	}

	size_t size() {
		return n * sizeof(T);
	}

	cPtr<T> ptr() { return p; }

	T *raw_ptr() { return p.raw_ptr(); }

	/* "decay" to pointer when operating on an array
	* e.g. for a vector v,   you can do set(v + (N-1), host_ptr, 1) instead of set(v.ptr() + (N-1),...)
	*/
	cPtr<T> operator+(size_t i) {return p + i;}
	cPtr<T> operator-(size_t i) {return p - i;}

	/* TODO: overload other operator*/
};



} /* namespace cu */
