#ifndef CUDA_TENSOR_WMMA_HPP
#define CUDA_TENSOR_WMMA_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// the current code assumes warp size is 32 threads for wmma operations
// this has held for as long as I can remember
// should future architectures break this 
// then the code may need adjustment
#ifndef WARP_SIZE
#define WARP_SIZE 32
#else
#if (WARP_SIZE != 32)
#error WARP_SIZE != 32
#endif
#endif

// non-looped hardcoded array copy of float4
template<int count> __device__ inline void _copy(const float4* src, float4* dst, const int srcstride = 1, const int dststride = 1);
template<> __device__ inline void _copy<1>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = src[0]; }
template<> __device__ inline void _copy<2>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = src[0]; dst[1*dststride] = src[1*srcstride]; }
template<> __device__ inline void _copy<3>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = src[0]; dst[1*dststride] = src[1*srcstride]; dst[2*dststride] = src[2*srcstride]; }
template<> __device__ inline void _copy<4>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = src[0]; dst[1*dststride] = src[1*srcstride]; dst[2*dststride] = src[2*srcstride]; dst[3*dststride] = src[3*srcstride]; }

// non-looped hardcoded array copy of float4 using __ldg instruction (uses faster read-only cache)
template<int count> __device__ inline void _copyldg(const float4* src, float4* dst, const int srcstride = 1, const int dststride = 1);
template<> __device__ inline void _copyldg<1>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = __ldg(src+0); }
template<> __device__ inline void _copyldg<2>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = __ldg(src+0); dst[1*dststride] = __ldg(src+1*srcstride); }
template<> __device__ inline void _copyldg<3>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = __ldg(src+0); dst[1*dststride] = __ldg(src+1*srcstride); dst[2*dststride] = __ldg(src+2*srcstride); }
template<> __device__ inline void _copyldg<4>(const float4* src, float4* dst, const int srcstride, const int dststride)
{ dst[0] = __ldg(src+0); dst[1*dststride] = __ldg(src+1*srcstride); dst[2*dststride] = __ldg(src+2*srcstride); dst[3*dststride] = __ldg(src+3*srcstride); }

// function to modify an A-fragment into a B-fragment
template<typename T> __device__ inline void _adjust_atob(T& frag);

// fragment typedefs for row-major matrix used in A*B (with B=A^T) computation
template<int M = 16, int N = 16, int K = 16, typename VT = __half>
struct rowmat_frag_traits
{
	static const int m = M, n = N, k = K;
	typedef VT value_type;
	typedef wmma::fragment<wmma::matrix_a, M, N, K, VT, wmma::row_major> afrag_t;
	typedef wmma::fragment<wmma::matrix_b, M, N, K, VT, wmma::col_major> bfrag_t;
	
	// NOTE! we calculate corresponding float4 array size manually, [a/b]frag_t::num_elements is not accurate
	static const int afrag_float4_size = M * K * sizeof(VT) / (WARP_SIZE * sizeof(float4));
	static const int bfrag_float4_size = K * N * sizeof(VT) / (WARP_SIZE * sizeof(float4));
	static_assert(afrag_float4_size * WARP_SIZE * sizeof(float4) == M * K * sizeof(VT));
	static_assert(bfrag_float4_size * WARP_SIZE * sizeof(float4) == K * N * sizeof(VT));
	
	union afrag_u {
		afrag_t frag;                 // wmma fragment representation
		float4  w[afrag_float4_size]; // float4 representation
		// fill with a constant
		__device__ inline void fill(const VT& v) { wmma::fill_fragment(frag, v); }
		// load/store using fragment instructions (sub-optimal 64 bit reads)
		__device__ inline void load_frag(const VT* ptr, unsigned rowstride) { wmma::load_matrix_sync(frag, ptr, rowstride); }
		__device__ inline void store_frag(VT* ptr, unsigned rowstride) const { wmma::store_matrix_sync(ptr, frag, rowstride, wmma::mem_row_major); }
		// load/store using float4 instructions (better 128 bit reads)
		// NOTE! to be able to load/store as float4, this is using a special memory ordening of the matrix
	    __device__ inline void load_pack(const float4* ptr, const int f4stride = 1) { _copy<afrag_float4_size>(ptr, w, f4stride, 1); }
	    __device__ inline void loadc_pack(const float4* ptr, const int f4stride = 1) { _copyldg<afrag_float4_size>(ptr, w, f4stride, 1); }
	    __device__ inline void store_pack(float4* ptr, const int f4stride = 1) const { _copy<afrag_float4_size>(w, ptr, 1, f4stride); }
	};
	union bfrag_u {
		bfrag_t frag;                 // wmma fragment representation
		float4  w[bfrag_float4_size]; // float4 representation
		// fill with a constant
		__device__ inline void fill(const VT& v) { wmma::fill_fragment(frag, v); }
		// load/store using fragment instructions (sub-optimal 64 bit reads)
		__device__ inline void load_frag(const VT* ptr, unsigned rowstride) { wmma::load_matrix_sync(frag, ptr, rowstride); }
		__device__ inline void store_frag(VT* ptr, unsigned rowstride) const { wmma::store_matrix_sync(ptr, frag, rowstride, wmma::mem_col_major); }
		// load/store using float4 instructions (better 128 bit reads)
		// NOTE! to be able to load as float4, the matrix has to be reordered in memory.
	    __device__ inline void load_pack(const float4* ptr, const int f4stride = 1) { _copy<bfrag_float4_size>(ptr, w, f4stride, 1); }
	    __device__ inline void loadc_pack(const float4* ptr, const int f4stride = 1) { _copyldg<bfrag_float4_size>(ptr, w, f4stride, 1); }
	    __device__ inline void store_pack(float4* ptr, const int f4stride = 1) const { _copy<bfrag_float4_size>(w, ptr, 1, f4stride); }
		// load/store using float4 instructions (better 128 bit reads) 
		// NOTE! like the above 3 functions, these assume the matrix is reordered in memory
		// The above 3 functions assume the matrix is reordered for fast bfrag_u loading,
		// the below functions assume the matrix is reordered for fast afrag_u loading and makes the needed adjustments
	    __device__ inline void load_pack_froma(const float4* ptr, const int f4stride = 1) { _copy<bfrag_float4_size>(ptr, w, f4stride, 1); _adjust_atob(*this); }
	    __device__ inline void loadc_pack_froma(const float4* ptr, const int f4stride = 1) { _copyldg<bfrag_float4_size>(ptr, w, f4stride, 1); _adjust_atob(*this); }
	};
};

template<int M = 16, int N = 16, int K = 16, typename AT = float>
struct accfrag_traits
{
	static const int m = M, n = N, k = K;
	typedef AT value_type;

	typedef wmma::fragment<wmma::accumulator, M, N, K, AT> cfrag_t;

	// NOTE! we calculate corresponding float4 array size manually, frag_t::num_elements is not accurate
	static const int cfrag_float4_size = M * N * sizeof(AT) / (WARP_SIZE * sizeof(float4));
	static_assert(cfrag_float4_size * WARP_SIZE * sizeof(float4) == M * N * sizeof(AT));

	union cfrag_u {
		cfrag_t frag;                 // wmma fragment representation
		float4  w[cfrag_float4_size]; // float4 representation
		// fill with a constant
		__device__ inline void fill(const AT& v) { wmma::fill_fragment(frag, v); }
		// load/store using fragment instructions
		__device__ inline void load_frag(const AT* ptr, unsigned rowstride) { wmma::load_matrix_sync(frag, ptr, rowstride, wmma::mem_row_major); }
		__device__ inline void store_frag(AT* ptr, unsigned rowstride) const { wmma::store_matrix_sync(ptr, frag, rowstride, wmma::mem_row_major); }
		// load/store using float4 instructions (better 128 bit reads)
		// NOTE! to be able to load/store as float4, this is using a special memory ordening of the matrix
	    __device__ inline void load_pack(const AT* ptr, const int f4stride = 1) { _copy<cfrag_float4_size>(ptr, w, f4stride, 1); }
	    __device__ inline void store_pack(AT* ptr, const int f4stride = 1) const { _copy<cfrag_float4_size>(w, ptr, 1, f4stride); }
	};
};

// function to modify an A-fragment into a B-fragment
// specialisation for n=m=k=16 on NVIDIA TITAN RTX
// NOTE! may not work on other devices and other fragment configurations
template<> __device__ inline void _adjust_atob<rowmat_frag_traits<16,16,16,__half>::bfrag_u>
                                              (rowmat_frag_traits<16,16,16,__half>::bfrag_u& frag)
{
	// swap half 2:3 and 4:5 to have correct mapping
	float tmp = frag.w[0].y;
	frag.w[0].y = frag.w[0].z;
	frag.w[0].z = tmp;
}



// row-order matrix for one CUDA block
// dim=rowlen is template param : must be multiple of afrag width
// nr of rows is variable       : must be multiple of afrag height
template<int DIM, typename FT, bool REORDER = true>
struct row_matrix
{
	typedef FT fragment_traits;
	typedef typename FT::value_type value_type;
	typedef typename FT::afrag_u afrag_u;
	typedef typename FT::bfrag_u bfrag_u;

	static const int dim = DIM, M = FT::m, N = FT::n, K = FT::k;
	static const bool reorder_fastload = REORDER;
	static const int frag_height = K;
	static const int frag_width = M;
	static const int row_frags = DIM / frag_width;
	static const int afrag_float4_size = FT::afrag_float4_size;
	static const int bfrag_float4_size = FT::bfrag_float4_size;

	/* static checks */
	// dimension must be an integer multiple of afrag width
	static_assert(row_frags * M == DIM);
	
	// pointer to actual matrix & nr of rows
	value_type* ptr;
	int nrrows;
	
	__device__ row_matrix(value_type* _ptr, int _nrrows)
		: ptr(_ptr), nrrows(_nrrows)
	{
		// assume flat thread structure inside CUDA block
//		static_assert(threadIdx.y == 0);
//		static_assert(threadIdx.z == 0);

		// round nrrows down to multiple of afrag height
		nrrows = ((nrrows / K) * K);
	}
	
	__device__ inline value_type* row_ptr(int rowid) { return ptr + (rowid * dim); }
	__device__ inline const value_type* row_ptr(int rowid) const { return ptr + (rowid * dim); }
	
	/* below functions all need to be called by an entire warp */
	// single A-fragment loading functions
	inline __device__ void load_frag_frag(afrag_u* afrag, int rowid, int fragid) const
	{
		afrag->load_frag(ptr + (rowid*dim) + fragid*M, dim);
	}
	inline __device__ void load_frag_pack(afrag_u* afrag, int rowid, int fragid) const
	{
		afrag->load_pack((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + fragid*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void load_frag(afrag_u* afrag, int rowid, int fragid) const
	{
		if (reorder_fastload)
			load_frag_pack(afrag, rowid, fragid);
		else
			load_frag_frag(afrag, rowid, fragid);
	}
	// loadc variants that use __ldg
	inline __device__ void loadc_frag_pack(afrag_u* afrag, int rowid, int fragid) const
	{
        afrag->loadc_pack((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + fragid*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void loadc_frag(afrag_u* afrag, int rowid, int fragid) const
	{
		static_assert(reorder_fastload, "Can't loadc without reordering");
		loadc_frag_pack(afrag, rowid, fragid);
	}
	// single B-fragment loading functions (where B=A^T)
	inline __device__ void bload_frag_frag(bfrag_u* bfrag, int rowid, int fragid) const
	{
		bfrag->load_frag(ptr + (rowid*dim) + fragid*M, dim);
	}
	inline __device__ void bload_frag_pack(bfrag_u* bfrag, int rowid, int fragid) const
	{
		bfrag->load_pack_froma((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + fragid*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void bload_frag(bfrag_u* bfrag, int rowid, int fragid) const
	{
		if (reorder_fastload)
			bload_frag_pack(bfrag, rowid, fragid);
		else
			bload_frag_frag(bfrag, rowid, fragid);
	}
	// bloadc variants that use __ldg
	inline __device__ void bloadc_frag_pack(bfrag_u* bfrag, int rowid, int fragid) const
	{
        bfrag->loadc_pack_froma((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + fragid*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void bloadc_frag(bfrag_u* bfrag, int rowid, int fragid) const
	{
		static_assert(reorder_fastload, "Can't loadc without reordering");
		bloadc_frag_pack(bfrag, rowid, fragid);
	}


	/* below functions all need to be called by an entire warp */
	// row A-fragment loading/storing functions
	inline __device__ void load_row_frags_frag(afrag_u afrag[row_frags], int rowid)
	{
		#pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			afrag[i].load_frag(ptr + (rowid*dim) + i*M, dim);
	}
	inline __device__ void load_row_frags_pack(afrag_u afrag[row_frags], int rowid)
	{
		#pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			afrag[i].load_pack((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + i*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void load_row_frags(afrag_u afrag[row_frags], int rowid)
	{
		if (reorder_fastload)
			load_row_frags_pack(afrag, rowid);
		else
			load_row_frags_frag(afrag, rowid);
	}
	// loadc variants that use __ldg
	inline __device__ void loadc_row_frags_pack(afrag_u afrag[row_frags], int rowid)
	{
		#pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			afrag[i].loadc_pack((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + i*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void loadc_row_frags(afrag_u afrag[row_frags], int rowid)
	{
		static_assert(reorder_fastload, "Can't loadc without reordering");
		loadc_row_frags_pack(afrag, rowid);
	}
	// store
	inline __device__ void store_row_frags_frag(afrag_u afrag[row_frags], int rowid)
	{
		#pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			afrag[i].store_frag(ptr + (rowid*dim) + i*M, dim);
	}
	inline __device__ void store_row_frags_pack(afrag_u afrag[row_frags], int rowid)
	{
		#pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			afrag[i].store_pack((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + i*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void store_row_frags(afrag_u afrag[row_frags], int rowid)
	{
		if (reorder_fastload)
			store_row_frags_pack(afrag, rowid);
		else
			store_row_frags_frag(afrag, rowid);
	}
	// row B-fragment loading functions (where B=A^T)
	inline __device__ void bload_row_frags_frag(bfrag_u bfrag[row_frags], int rowid)
	{
		#pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			bfrag[i].load_frag(ptr + (rowid*dim) + i*M, dim);
	}
	inline __device__ void bload_row_frags_pack(bfrag_u bfrag[row_frags], int rowid)
	{
        #pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			bfrag[i].load_pack_froma((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + i*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void bload_row_frags(bfrag_u bfrag[row_frags], int rowid)
	{
		if (reorder_fastload)
			bload_row_frags_pack(bfrag, rowid);
		else
			bload_row_frags_frag(bfrag, rowid);
	}
	// bloadc variants that use __ldg
	inline __device__ void bloadc_row_frags_pack(bfrag_u bfrag[row_frags], int rowid)
	{
		#pragma unroll
		for (unsigned i = 0; i < row_frags; ++i)
			bfrag[i].loadc_pack_froma((float4*)(ptr + (rowid*dim)) + (threadIdx.x % WARP_SIZE) + i*WARP_SIZE*afrag_float4_size, WARP_SIZE);
	}
	inline __device__ void bloadc_row_frags(bfrag_u bfrag[row_frags], int rowid)
	{
		static_assert(reorder_fastload, "Can't loadc without reordering");
		bloadc_row_frags_pack(bfrag, rowid);
	}
	
	// to be called by a single entire CUDA block
	// reorders matrix for faster loading of wmma fragments
	__device__ void prepare_fastload()
	{
		if (!reorder_fastload)
			return;
		
		// each warp processes frag_height rows at a time by its own

		afrag_u afrag[row_frags];

		for (int rowid = (threadIdx.x/WARP_SIZE) * frag_height; rowid < nrrows; rowid += (blockDim.x/WARP_SIZE) * frag_height)
		{
			load_row_frags_frag(afrag, rowid);
			store_row_frags_pack(afrag, rowid);
		}
		
        // synchronize all threads in the block, and ensure all writes to shared are visible
        __syncthreads();
        __threadfence_block();
	}
	
	// to be called by entire CUDA block
	// undo reordering matrix for faster loading of wmma fragments
	__device__ void undo_fastload()
	{
		if (!reorder_fastload)
			return;

		// each warp processes frag_height rows at a time

		afrag_u afrag[row_frags];
		
		for (int rowid = (threadIdx.x/WARP_SIZE) * frag_height; rowid < nrrows; rowid += (blockDim.x/WARP_SIZE) * frag_height)
		{
			load_row_frags_pack(afrag, rowid);
			store_row_frags_frag(afrag, rowid);
		}

        // synchronize all threads in the block, and ensure all writes to shared are visible
        __syncthreads();
        __threadfence_block();
	}

    __device__ void prepare_fastload_partial(int from, int to)
	{
		if (!reorder_fastload)
			return;
		
		// each warp processes frag_height rows at a time by its own

		afrag_u afrag[row_frags];

		for (int rowid = from + (threadIdx.x/WARP_SIZE) * frag_height; rowid < to; rowid += (blockDim.x/WARP_SIZE) * frag_height)
		{
			load_row_frags_frag(afrag, rowid);
			store_row_frags_pack(afrag, rowid);
		}
		
        // synchronize all threads in the block, and ensure all writes to shared are visible
        __syncthreads();
        __threadfence_block();
	}


};

// row_matrix_cache
// 1. load reg_rowblocks into registerspace of entire block
// 2. repeatedly copy rowblock from registerspace into shared memory & process
// interleave operations to ensure optimal performance:
// - load new rowblock when that part of registerspace will not be used anymore in prior cycle
// - use at least 2 rowblocks in shared memory and do: copy1, [copy0+process1 , copy1+process0, repeat ...]
template<typename row_mat, int _reg_rowblocks, int _shared_rowblocks, int warps_per_block>
struct row_matrix_cache
{
	typedef row_mat row_matrix_t;
	typedef typename row_matrix_t::value_type value_type;
	static const int shared_rowblocks = _shared_rowblocks;
	static const int reg_rowblocks = _reg_rowblocks;
	static const int row_frags = row_matrix_t::row_frags;
	static const int frag_height = row_matrix_t::frag_height;
	static const int reg_rows = reg_rowblocks * frag_height;
	static const int afrag_float4_size = row_matrix_t::afrag_float4_size;
	static const int cache_regs = (reg_rowblocks * row_frags * afrag_float4_size + warps_per_block - 1) / warps_per_block;

	typedef float4 shared_cache_t[shared_rowblocks][row_frags * afrag_float4_size * WARP_SIZE];
	//__shared__ shared_cache_t shared_cache; // needs to be defined external to this struct, e.g. in the kernel function

	float4 reg_cache[cache_regs];



	// call all member functions with entire block with identical parameter
	// please make sure to call these inlined functions with compile time constant inputs
	// to avoid computing pointers too much
	// e.g. in a loop with #pragma unroll
	
	__device__ inline void load_reg_rowblocks(const row_matrix_t& matrix, const int row_id)
	{
		const float4* ptr = (float4*)matrix.row_ptr(row_id);
		const float4* ptrend = (float4*)matrix.row_ptr(row_id + frag_height * reg_rowblocks);
		ptr += threadIdx.x;
		#pragma unroll
		for (int i = 0; i < cache_regs - 1; ++i, ptr += (WARP_SIZE * warps_per_block)/*=blockDim.x*/ )
			reg_cache[i] = __ldg(ptr);
		if (ptr < ptrend)
			reg_cache[cache_regs - 1] = __ldg(ptr);
	}
	
	__device__ inline void store_shared_rowblocks(float4* shared_cache, const int rowblock_id)
	{
		const int start_reg = (rowblock_id * row_frags * afrag_float4_size) / warps_per_block;
		const int end_reg = start_reg + (row_frags * afrag_float4_size + warps_per_block - 1) / warps_per_block;
		// TODO check formula
		float4* ptr2 = shared_cache + ((start_reg * warps_per_block) - (rowblock_id * row_frags * afrag_float4_size))*WARP_SIZE + threadIdx.x;
		float4* ptrend = shared_cache + (row_frags * afrag_float4_size * WARP_SIZE);
		if (ptr2 >= shared_cache)
			*ptr2 = reg_cache[start_reg];
		ptr2 += (WARP_SIZE * warps_per_block);
		#pragma unroll
		for (int i = start_reg+1; i < end_reg-1; ++i)
		{
			*ptr2 = reg_cache[i];
			ptr2 += (WARP_SIZE * warps_per_block);
		}
		if (end_reg-1 != start_reg && ptr2 < ptrend)
			*ptr2= reg_cache[end_reg-1];
	}
};


// row-order data for one CUDA block
// nr of rows is variable       : must be multiple of afrag height
template<typename FT>
struct row_data
{
	typedef FT value_type;

	// pointer to actual data & nr of rows
	const value_type* ptr;
	int nrrows;
	
	__device__ row_data(const value_type* _ptr, int _nrrows)
		: ptr(_ptr), nrrows(_nrrows)
	{
		// assume flat thread structure inside CUDA block
//		static_assert(threadIdx.y == 0);
//		static_assert(threadIdx.z == 0);

		// round nrrows down to multiple of afrag height
		nrrows = ((nrrows / 16) * 16);
	}
	
	__device__ inline value_type* row_ptr(int rowid) { return ptr + (rowid); }
	__device__ inline const value_type* row_ptr(int rowid) const { return ptr + (rowid); }
};

template<typename row_data, int _rows, int warps_per_block, bool a_side=true>
struct row_data_shared
{
	typedef row_data row_data_t;
	typedef typename row_data_t::value_type value_type;
	static const int rows = _rows;
    static const int rows_per_float = sizeof(float) / sizeof(value_type);

    // for now we only support half
    static_assert( sizeof(row_data_t::value_type) == 2);

	typedef value_type shared_cache_t[ rows ];
	//__shared__ shared_cache_t shared_cache; // needs to be defined external to this struct, e.g. in the kernel function

	// call all member functions with entire block with identical parameter
	// please make sure to call these inlined functions with compile time constant inputs
	// to avoid computing pointers too much
	// e.g. in a loop with #pragma unroll
	
	__device__ inline half2 load_fragment_data(const value_type* shared_cache, const int rowblock_id)
	{
		static_assert(a_side);
        return *(reinterpret_cast<const half2*>(shared_cache + 16*rowblock_id) + (threadIdx.x%32)/8 + 4*((threadIdx.x%8) > 3));
	}


    __device__ inline half2 bload_fragment_data(const value_type* shared_cache, const int rowblock_id, const int shift) {
        static_assert(!a_side);
        return *(reinterpret_cast<const half2*>(shared_cache + 16*rowblock_id + 2 * (threadIdx.x%4) + 8 * shift));
    }

    // fill shared cache with global values
	__device__ inline void global_to_shared_rows(const row_data_t& data, value_type* shared_cache, const int row_id)
	{
	    const float* ptr = (float*)data.row_ptr(row_id);
        const float* ptrend = (float*)(data.row_ptr(row_id + rows));
        ptr += threadIdx.x;
        #pragma unroll
        for(int i = rows_per_float*threadIdx.x; i < rows; i+= rows_per_float * WARP_SIZE * warps_per_block, ptr += (WARP_SIZE*warps_per_block)) {
           *reinterpret_cast<float*>(shared_cache + i) = __ldg(ptr);
        }

        if( a_side ) {
            __syncthreads();

            // reorder and half size for easy retrieval related to fragments
            // assumes #threads >= #rows/4
            const int a1 = 16 * (threadIdx.x / 4) + 2*(threadIdx.x%4);
            const int a2 = a1 + 8;
            if( a1 < rows ) {
                half2 d1 = *reinterpret_cast<half2*>(shared_cache+a1);
                half2 d2 = *reinterpret_cast<half2*>(shared_cache+a2);
                // swap second of d1 with first of d2
                half tmp = d1.y;
                d1.y = d2.x;
                d2.x = tmp;
                *reinterpret_cast<half2*>(shared_cache+a1) = d1;
                *reinterpret_cast<half2*>(shared_cache+a2) = d2;
            }
        }
	}
};

#endif
