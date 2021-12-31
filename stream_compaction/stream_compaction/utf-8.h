#pragma once

namespace UTF8 {
	typedef uint8_t byte;
	typedef uint32_t rune;


	/* decodes a string 'in' of UTF-8 characters of length n, storing the output in out
	 * out must be large enough to contain the output (if unknown, 4*n)
	 * 'in' must be a valid UTF-8 string
	 */
	size_t cpu_decode(const byte *in, size_t n, rune *out);

	/* decodes a string 'in' of UTF-8 characters of length n, storing the output in out
	 * out must be large enough to contain the output (if unknown, 4*n)
	 * parallelizes on the GPU
	 * 'in' must be a valid UTF-8 string
	 * returns number of code-points decoded
	 */
	size_t decode(const byte *in, size_t n, rune *out, size_t out_len);
}
