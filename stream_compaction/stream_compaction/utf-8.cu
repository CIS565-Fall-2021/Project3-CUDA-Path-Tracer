#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "cVec.h"
#include "utf-8.h"

#define block_size 128

namespace UTF8 {

	/* Format of UTF-8:
	 * Each unicode code-point is encoded with one to four bytes, whose forms are:
	 * 1	0xxxxxxx					U+0000  to  U+007F
	 * 2	110xxxxx 10xxxxxx				U+0080  to  U+07FF
	 * 3	1110xxxx 10xxxxxx 10xxxxxx			U+0800  to  U+FFFF
	 * 4	11110xxx 10xxxxxx 10xxxxxx 10xxxxxx		U+10000 to  U+10FFFF
	 * UTF-8 is a superset of ASCII, such that code-points of length 1 correspond
	 * precisely to ASCII characters.
	 * Valid UTF-8 must be a sequence of these code-points, all other bytes (or
	 * out-of-place bytes) are invalid. Code points above U+10FFFF, even if they
	 * can be represented by the representation above are invalid as of RFC3629.
	 * 
	 * When decoding, the "x" bytes are concatenated into a rune of 4 bytes.
	 * For example, the UTF-8 character 0xEFBFBD (U+FFFD) is converted to 0xFFFD.
	 * Encoding runes to UTF-8 reverses this operation.
	 */


	/* decodes a string 'in' of UTF-8 characters of length n bytes, storing the output in 'out'
	 * 'out' must be large enough to contain the output
	 * 'in' must be a valid UTF-8 string
	 * returns the number of code-points decoded
	 */
	size_t cpu_decode(const byte *in, size_t n, rune *out)
	{
		const byte *s;
		int count;

		for (s = in, count = 0; s < in + n; count++, out++) {
			if ((*s & 0x80) == 0) { /* 1-byte code-point */
				*out = (rune) s[0];
				s++;
			} else if ((*s & 0xe0) == 0xc0) { /* 2-byte code-point*/
				*out = ((rune) (s[0] & 0x1f) << 6) | /* bytes from 110xxxxx */
				       ((rune) (s[1] & 0x3f));       /* bytes from 10xxxxxx */
				s += 2;
			} else if ((*s & 0xf0) == 0xe0) { /* 3-byte code-point */
				*out = ((rune) (s[0] & 0x0f) << 12) | /* bytes from 1100xxxx */
				       ((rune) (s[1] & 0x3f) <<  6) | /* bytes from 10xxxxxx */
				       ((rune) (s[2] & 0x3f));        /* bytes from 10xxxxxx */
				s += 3;
			} else {
				/* this implementation assumes the encoding is valid and performs no checks here */
				*out = ((rune) (s[0] & 0x07) << 18) | /* bytes from 11110xxxx */
				       ((rune) (s[1] & 0x3f) << 12) | /* bytes from 10xxxxxxx */
				       ((rune) (s[2] & 0x3f) <<  6) | /* bytes from 10xxxxxxx */
				       ((rune) (s[3] & 0x3f));        /* bytes from 10xxxxxxx */
				s += 4;
			}
		}
		return count;
	}

	/* encodes a list of n code-points from 'in', storing the output in 'out'
	 * 'out' must be large enough to contain the output
	 * 'in' must be a sequence of valid UTF-8 code-points (U+0000 to U+10FFFF or 0x0000 to 0x10FFFF)
	 * returns the number of bytes written to output
	 * FIXME: this implementation is incomplete and incorrect
	 */
	size_t cpu_encode(const rune *in, size_t n, byte *out)
	{
		const rune *s;
		int count;

		for (s = in, count = 0; s < in + n; s++, count++) {
			if (*s < 0x80) { /* 1-byte code point */
				*out++ = (byte) (*s & 0x8f);
			} else if (*s < 0x800) {
				*out++ = (byte) (((*s >> 6) & 0x1f) | 0xc0); /* bytes for 110xxxxx */
				*out++ = (byte) (*s & 0x3f) | 0x80;          /* bytes for 10xxxxxx */
			} else if (*s < 0x10000) {
				*out++ = (byte) (((*s >> 12) & 0x3f) | 0x0);
			}
		}
		return count;
	}
	

	__constant__ int offset_lookup[32] = { /* this array is used by kern_utf8_offsets to prevent branching. it is indexed by the top 5 bits of the UTF-8 byte*/
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0xxxxxxx */
		0, 0, 0, 0, 0, 0, 0, 0,				/* 10xxxxxx */
		2, 2, 2, 2,					/* 110xxxxx */
		3, 3,						/* 1110xxxx */
		4,						/* 11110xxx */
		0						/* invalid byte */ /* FIXME: these are lengths not offsets. e.g. 0xxxxxxx has an offset of 4 from prev code-point b/c the characters has a length of 1 bytes*/
	};

	/* for each byte of input, writes the relative offset it should be placed in in the expanded UTF-8
	 * 0xxxxxxx -> 4   10xxxxxx -> 1   110xxxxx -> 3   1110xxxx -> 2   11110xxx -> 1
	 * For example the UTF-8 input code-points       [110xxxxx 10xxxxxx] [0xxxxxxx] [1110xxxx 10xxxxxx 10xxxxxx]
	 * is mapped to [3 1] [4] [2 1 1]
	 * which corresponds to their offsets from the previous input byte (first offset is taken from -1) in the expanded representation:
	 * [00000000 00000000 110xxxxx 10xxxxxx] [00000000 00000000 00000000 0xxxxxxx] [00000000 1110xxxx 10xxxxxx 10xxxxxx]
	 * (the relative offsets are intended to be scanned to produce absolute offsets (with respect to position -1))
	 */
	__global__ void kern_utf8_offsets(const byte* in, size_t n, size_t* offsets)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;						// FIXME  i thinj this needs to calc both offset and length. or msybe just length and use length to calc offset in scatter
		
		offsets[idx] = offset_lookup[in[idx] >> 3];
	}

	/* expands each code-point to 4 bytes, writing them to the appropriate location in out*/
	__global__ void kern_utf8_scatter(const byte * in, size_t n, const size_t * offsets, byte* out)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;

		out[offsets[idx]-1] = in[idx]; /* global offsets are with respect to -1 */
	}


	/* decodes a string 'in' of UTF-8 characters of length n bytes, storing the output in 'out' of length out_len runes
	 * out_len must be large enough to contain the output (if unknown, set to n)
	 * parallelizes on the GPU
	 * 'in' must be a valid UTF-8 string
	 * returns number of code-points decoded
	 */
	size_t decode(const byte *in, size_t n, rune *out, size_t out_len)
	{
		int log2n = ilog2ceil(n+1);
		int N = 1 << log2n; /* a power of 2 that can contain n+1 is necessary for scanning over offsets */

		cu::cVec<byte> dev_in(n, in);
		cu::cVec<size_t> offsets(N); /* because scan is exclusive, offsets[1:] is used after scan*/
		
		int block_count = ((n + block_size - 1) / block_size);

		kern_utf8_offsets<<<block_count, block_size>>>(dev_in.raw_ptr(), n, offsets.raw_ptr());
		StreamCompaction::Efficient::scan_dev(N, &offsets);

		size_t length;
		cu::copy(&length, offsets + n, 1);
		length /= 4; /* number of code-points */

		cu::cVec<byte> dev_expanded(length * 4); /* 4-byte expansions represented as a sequence of bytes */
		cu::set<byte>(dev_expanded.ptr(), 0, length * 4);

		kern_utf8_scatter<<<block_count, block_size>>>(dev_in.raw_ptr(), n, offsets.raw_ptr() + 1, dev_expanded.raw_ptr());
		

		cu::cVec<rune> dev_out(length);
		cu::copy(out, dev_out.ptr(), length);

		return length;
	}
}
