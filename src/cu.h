#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda.h>


void cu_check_error_fn(const char *msg, const char *file, int line)
{
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err == cudaError::cudaSuccess)
		return;

	fprintf(stderr, "CUDA error");
	if (file)
		fprintf(stderr, " (%s:%d)", file, line);

	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	(void) getchar(); /* explicitly ignore result */
#  endif
	exit(EXIT_FAILURE);
}

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define cu_check_err(msg) cu_check_error_fn(msg, FILENAME, __LINE__)


template <typename T>
__host__ __device__ T max(T v) {
	return v;
}

template <typename T, typename... U>
__host__ __device__ T max(T v1, T v2, U ... vs) {
	return max(v1 > v2 ? v1 : v2, vs...);
}

template <typename T>
__host__ __device__ T min(T v) {
	return v;
}

template <typename T, typename... U>
__host__ __device__ T min(T v1, T v2, U ... vs) {
	return max(v1 < v2 ? v1 : v2, vs...);
}



namespace cu {

template <typename T> class cPtr;
template <typename T> class cVec;

template <typename T>
/* allocates device array T[len] of length len */
cPtr<T> make(size_t len)
{
	T *d;
	cudaMalloc((void **) &d, sizeof(*d) * len);
	cu_check_err("cu: cudaMalloc failed!");
	return cPtr<T>(d);
}

template <typename T>
/* deletes device array */
static void del(cPtr<T> p)
{
	cudaFree(p.get());
	cu_check_err("cu: cudaFree failed!");
}


/* a "raw" pointer for device memory, points to cuda allocated memory, not a smart pointer */
template <typename T>
class cPtr {
	T *p;
public:

	constexpr cPtr() : p(nullptr) {}
	cPtr(T *dev_ptr) : p(dev_ptr) {}
	cPtr(const cPtr<T> &cp) : p(cp.p) {}
	cPtr<T> &operator=(const cPtr<T> &cp) { p = cp.p; return *this; }
	cPtr(cPtr &&cp) : p(cp.p) { cp.p = nullptr; }
	cPtr &operator=(cPtr &&cp) { p = cp.p; cp.p = nullptr; return *this; }
	~cPtr() {}

	cPtr<T> &operator=(const cVec<T> &v) { p = v.p; return *this; }
	cPtr<T> &operator=(T *ptr) { p = ptr; return *this; }

	cPtr &operator++() { p++; return *this; }
	cPtr operator++(int) { cPtr<T> cp = *this; p++; return cp; }
	cPtr &operator--() { p--; return *this; }
	cPtr operator--(int) { cPtr<T> cp = *this; p--; return cp; }
	cPtr &operator+=(size_t i) { p += i; return *this; }
	cPtr &operator-=(size_t i) { p -= i; return *this; }
	cPtr operator+(size_t i) const { cPtr<T> cp = *this; cp.p += i; return cp; }
	cPtr operator-(size_t i) const { cPtr<T> cp = *this; cp.p -= i; return cp; }

	ptrdiff_t operator-(const cPtr<T> &b) const { return p - b.p; }

	bool operator!() const { return p != nullptr; }

	bool operator==(const cPtr<T> &b) const { return p == b.p; }
	bool operator!=(const cPtr<T> &b) const { return p != b.p; }
	bool operator<=(const cPtr<T> &b) const { return p <= b.p; }
	bool operator>=(const cPtr<T> &b) const { return p >= b.p; }
	bool operator< (const cPtr<T> &b) const { return p < b.p; }
	bool operator> (const cPtr<T> &b) const { return p > b.p; }

	T *get() { return p; }
	const T *get() const { return p; }
};

/* memsets a region of length of n elements to the byte val (each byte in region set to val) */
template <typename T>
void set(cPtr<T> dst, int val, size_t n = 1)
{
	cudaMemset(dst.get(), val, n * sizeof(T));
	cu_check_err("cuda memset failed!");
}

/* copies n elements from device pointer src to device pointer dst */
template <typename T>
void copy(cPtr<T> dst, const cPtr<T> src, size_t n)
{
	cudaMemcpy(dst.get(), src.get(), n * sizeof(T), cudaMemcpyDeviceToDevice);
	cu_check_err("cuda memcpy device to device failed!");
}

/* copies n elements from host pointer src to device pointer dst */
template <typename T>
void copy(cPtr<T> dst, const T *src, size_t n)
{
	cudaMemcpy(dst.get(), src, n * sizeof(T), cudaMemcpyHostToDevice);
	cu_check_err("cuda memcpy host to device failed!");
}

/* copies n elements from device pointer src to host pointer dst */
template <typename T>
void copy(T *dst, const cPtr<T> src, size_t n)
{
	cudaMemcpy(dst, src.get(), n * sizeof(T), cudaMemcpyDeviceToHost);
	cu_check_err("cuda memcpy device to host failed!");
}

/* copies n elements from host pointer src to host pointer dst */
template <typename T>
void copy(T *dst, const T *src, size_t n)
{
	cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToHost);
	cu_check_err("cuda memcpy host to host failed!");
}




/* smart container for manually allocated CUDA memory, holds both single vars and arrays */

template <typename T>
class cVec {
	size_t n;
	cPtr<T> p;

public:
	constexpr cVec() : n(0), p(nullptr) {}


	/* constructor, takes in host pointer, length of pointer array */
	cVec(size_t len, const T *host_ptr) : n(len)
	{
		p = make<T>(n);
		copy(p, host_ptr, n);
	}

	/* constructor, takes in host pointer, length of pointer array, and desired length of device array.  dev_len cannot be shorter than host_len */
	cVec(size_t host_len, const T *host_ptr, size_t dev_len) : n(dev_len)
	{
		p = make<T>(n);
		copy(p, host_ptr, host_len);
	}

	/* constructor, does not zero memory */
	cVec(size_t len) : n(len)
	{
		p = make<T>(n);
	}

	/* copy constructor */
	cVec(const cVec<T> &v) : n(v.n)
	{
		p = make<T>(n);
		copy(p, v.p, n);
	}

	/* copy assignment */
	cVec<T> &operator=(const cVec<T> &v)
	{
		if (v.n != n) {
			n = v.n;
			del(p);
			p = make<T>(n);
		}
		copy(p, v.p, n);
	}

	/* destructor */
	~cVec()
	{
		del(p);
		n = 0;
		p = nullptr;
	}

	/* move constructor */
	cVec(cVec &&v) : n(v.n), p(v.p)
	{
		v.n = 0;
		v.p = nullptr;
	}

	/* move assignment */
	cVec &operator=(cVec &&v)
	{
		del(p);
		n = v.n;
		p = v.p;
		v.n = 0;
		v.p = nullptr;
		return *this;
	}

	size_t length()
	{
		return n;
	}

	size_t size()
	{
		return n * sizeof(T);
	}

	cPtr<T> ptr() { return p; }

	T *get() { return p.get(); }

	/* "decay" to pointer when operating on an array
	* e.g. for a vector v,   you can do set(v + (N-1), host_ptr, 1) instead of set(v.ptr() + (N-1),...)
	*/
	cPtr<T> operator+(size_t i) { return p + i; }
	cPtr<T> operator-(size_t i) { return p - i; }

	/* TODO: overload other operator*/
};



} /* namespace cu */
