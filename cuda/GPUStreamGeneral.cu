#include <iostream>
#include <stdexcept>
#include <numeric>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <immintrin.h>
#include <iterator>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "cuda_tensor_wmma.hpp"
#include <curand_kernel.h>

#include "GPUStreamGeneral.h"

using namespace nvcuda;

#define CUDA_CHECK(s) \
    { \
        auto s_ret = s; \
        if ( s_ret != cudaSuccess ) \
        { \
            const char* errorstr = cudaGetErrorString(s_ret); \
            std::cerr << "CUDA Error (" << __FILE__ << "@l" << __LINE__ << "): " << errorstr << std::endl; \
            throw std::runtime_error(errorstr); \
        } \
    }

#ifndef GPUVECNUM
#define GPUVECNUM 65536
#endif

#define VECNUM GPUVECNUM

// Fully coalesced memory loads and stores
// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_float_to_8bit( const float* Y, dhtype* Yint8, const uint32_t DIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;

    const uint32_t Ystart = ROWS_PER_BLOCK * DIM * blockIdx.x;

    float4* Yptr = ((float4*)(Y + Ystart)) + threadIdx.x;
    const float4* Yptr_end = (float4*)(Y + Ystart + ROWS_PER_BLOCK * DIM);

    int* Yint8ptr = (int*)(Yint8 + Ystart) + threadIdx.x;

    __syncthreads();

    for( ; Yptr < Yptr_end; Yptr += WARP_SIZE, Yint8ptr += WARP_SIZE) {
        float regs[4];
        dhtype regs_out[4];
        *reinterpret_cast<float4*>(regs) = *Yptr;

        #pragma unroll 
        for( uint32_t i = 0; i < 4; i++ )
            regs_out[i] = dhtype(__float2uint_rn((regs[i] - floorf(regs[i]))*256.));

        *Yint8ptr = *reinterpret_cast<int*>( regs_out );
    }
}

// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
template< uint32_t DIM>
__global__ void kernel_float_to_8bit_transpose( const float* Y, dhtype* Yint8 ) {
    const uint32_t ROWS_PER_BLOCK = 32;

    const uint32_t Ystart = ROWS_PER_BLOCK * DIM * blockIdx.x;

    float4* Yptr = ((float4*)(Y + Ystart)) + threadIdx.x;
    const float4* Yptr_end = (float4*)(Y + Ystart + ROWS_PER_BLOCK * DIM);

    int* Yint8ptr = (int*)(Yint8 + Ystart) + threadIdx.x;

    __shared__ uint32_t out_block[DIM/4][32];

    __syncthreads();

    uint32_t ind = threadIdx.x;
    for( ; Yptr < Yptr_end; Yptr += WARP_SIZE, ind += WARP_SIZE) {
        float regs[4];
        dhtype regs_out[4];
        *reinterpret_cast<float4*>(regs) = *Yptr;

        #pragma unroll 
        for( uint32_t i = 0; i < 4; i++ )
            regs_out[i] = dhtype(__float2uint_rn((regs[i] - floorf(regs[i]))*256.));

        out_block[ind%(DIM/4)][ind/(DIM/4)] = *reinterpret_cast<uint32_t*>( regs_out );
    }

    __syncthreads();
    for( uint32_t i = 0; i < DIM/4; i++, Yint8ptr += WARP_SIZE ) {
        *Yint8ptr = out_block[i][threadIdx.x];
    }
}

// Call with vecs/ROWS_PER_BLOCK blocks with 32 threads each
__global__ void __launch_bounds__(32) X_to_Xfloat( const int16_t* X, float* Xfloat, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;
    const uint32_t SHORTS_PER_FLOAT2 = 4;

    float2* Xptr = ((float2*)(X + ROWS_PER_BLOCK * VECDIM * blockIdx.x)) + threadIdx.x;
    const float2* Xptr_end = (float2*)(X + ROWS_PER_BLOCK * VECDIM * (blockIdx.x+1));
    float4* Xfloatptr = ((float4*)(Xfloat + ROWS_PER_BLOCK * VECDIM * blockIdx.x)) + 2*threadIdx.x;
    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xfloatptr += WARP_SIZE ) {
        int16_t regs[SHORTS_PER_FLOAT2];
        float regs_out[SHORTS_PER_FLOAT2];
        *reinterpret_cast<float2*>(regs) = *Xptr;
        
        #pragma unroll 
        for( uint32_t i = 0; i < SHORTS_PER_FLOAT2; i++ )
            regs_out[i] = float( regs[i] );

        *Xfloatptr = *reinterpret_cast<float4*>( regs_out );
    }
}


// Converts X to Xhalf and negates depending on the sign of the corresponding ip
// Fully coalesced memory loads and stores
// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_X_to_Xhalf_negate( const int16_t* X, const half* ips, half* Xhalf, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;
    const uint32_t SHORTS_PER_FLOAT4 = 8;

    const uint32_t Xstart = ROWS_PER_BLOCK * VECDIM * blockIdx.x;

    float4* Xptr = ((float4*)(X + Xstart)) + threadIdx.x;
    const float4* Xptr_end = (float4*)(X + Xstart + ROWS_PER_BLOCK * VECDIM);
    float4* Xhalfptr = ((float4*)(Xhalf + Xstart)) + threadIdx.x;

    __shared__ bool s_sign[ROWS_PER_BLOCK];
    s_sign[threadIdx.x] = __hgt( ips[ROWS_PER_BLOCK * blockIdx.x + threadIdx.x], half(0.));
    
    __syncthreads();

    uint32_t k = SHORTS_PER_FLOAT4*threadIdx.x;
    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xhalfptr += WARP_SIZE, k+=SHORTS_PER_FLOAT4*WARP_SIZE ) {
        int16_t regs[SHORTS_PER_FLOAT4];
        half regs_out[SHORTS_PER_FLOAT4];
        *reinterpret_cast<float4*>(regs) = *Xptr;

        half sign = ( s_sign[k/VECDIM] ) ? half(1.) : half(-1.);
        #pragma unroll 
        for( uint32_t i = 0; i < SHORTS_PER_FLOAT4; i++ )
            regs_out[i] = sign * __short2half_rn( regs[i] );

        *Xhalfptr = *reinterpret_cast<float4*>( regs_out );
    }
}

// Converts X to Xhalf and normalizes the length
// Fully coalesced memory loads and stores
// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_X_to_Xhalf_normalize( const int16_t* X, const lentype* Xlen, half* Xhalf, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;
    const uint32_t SHORTS_PER_FLOAT4 = 8;

    const uint32_t Xstart = ROWS_PER_BLOCK * VECDIM * blockIdx.x;

    float4* Xptr = ((float4*)(X + Xstart)) + threadIdx.x;
    const float4* Xptr_end = (float4*)(X + Xstart + ROWS_PER_BLOCK * VECDIM);
    float4* Xhalfptr = ((float4*)(Xhalf + Xstart)) + threadIdx.x;

    __shared__ half s_len[ROWS_PER_BLOCK];
    s_len[threadIdx.x] = __float2half( Xlen[ROWS_PER_BLOCK * blockIdx.x + threadIdx.x]);
    
    __syncthreads();

    uint32_t k = SHORTS_PER_FLOAT4*threadIdx.x;
    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xhalfptr += WARP_SIZE, k+=SHORTS_PER_FLOAT4*WARP_SIZE ) {
        int16_t regs[SHORTS_PER_FLOAT4];
        half regs_out[SHORTS_PER_FLOAT4];
        *reinterpret_cast<float4*>(regs) = *Xptr;

        half normalize_factor = half( rsqrtf( s_len[k/VECDIM] ) );
        #pragma unroll 
        for( uint32_t i = 0; i < SHORTS_PER_FLOAT4; i++ )
            regs_out[i] = normalize_factor * __short2half_rn( regs[i] );

        *Xhalfptr = *reinterpret_cast<float4*>( regs_out );
    }
}


// Converts X to Xhalf
// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_X_to_Xhalf( const int16_t* X, half* Xhalf, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;
    const uint32_t SHORTS_PER_FLOAT4 = 8;

    float4* Xptr = ((float4*)(X + ROWS_PER_BLOCK * VECDIM * blockIdx.x)) + threadIdx.x;
    const float4* Xptr_end = (float4*)(X + ROWS_PER_BLOCK * VECDIM * (blockIdx.x+1));
    float4* Xhalfptr = ((float4*)(Xhalf + ROWS_PER_BLOCK * VECDIM * blockIdx.x)) + threadIdx.x;

    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xhalfptr += WARP_SIZE ) {
        int16_t regs[SHORTS_PER_FLOAT4];
        half regs_out[SHORTS_PER_FLOAT4];
        *reinterpret_cast<float4*>(regs) = *Xptr;
        
        #pragma unroll 
        for( uint32_t i = 0; i < SHORTS_PER_FLOAT4; i++ )
            regs_out[i] = __short2half_rn( regs[i] );

        *Xhalfptr = *reinterpret_cast<float4*>( regs_out );
    }
}

// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_float_to_half( const float* in, half* out, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;

    float4* Xptr = ((float4*)(in + ROWS_PER_BLOCK * VECDIM * blockIdx.x)) + threadIdx.x;
    const float4* Xptr_end = (float4*)(in + ROWS_PER_BLOCK * VECDIM * (blockIdx.x+1));
    float2* Xhalfptr = ((float2*)(out + ROWS_PER_BLOCK * VECDIM * blockIdx.x)) + threadIdx.x;

    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xhalfptr += WARP_SIZE ) {
        float regs[4];
        half regs_out[4];
        *reinterpret_cast<float4*>(regs) = *Xptr;
        
        #pragma unroll 
        for( uint32_t i = 0; i < 4; i++ )
            regs_out[i] = __float2half( regs[i] );

        *Xhalfptr = *reinterpret_cast<float2*>( regs_out );
    }
}

// Converts X to Xfloat
// Fully coalesced memory loads and stores
// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_X_to_Xfloat( const int16_t* X, float* Xfloat, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;
    const uint32_t SHORTS_PER_FLOAT4 = 8;

    const uint32_t Xstart = ROWS_PER_BLOCK * VECDIM * blockIdx.x;

    float4* Xptr = ((float4*)(X + Xstart)) + threadIdx.x;
    const float4* Xptr_end = (float4*)(X + Xstart + ROWS_PER_BLOCK * VECDIM);
    float4* Xfloatptr = ((float4*)(Xfloat + Xstart)) + 2 * threadIdx.x;

    __syncthreads();

    uint32_t k = SHORTS_PER_FLOAT4*threadIdx.x;
    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xfloatptr += 2*WARP_SIZE, k+=SHORTS_PER_FLOAT4*WARP_SIZE ) {
        int16_t regs[SHORTS_PER_FLOAT4];
        float regs_out[SHORTS_PER_FLOAT4];
        *reinterpret_cast<float4*>(regs) = *Xptr;

        #pragma unroll 
        for( uint32_t i = 0; i < SHORTS_PER_FLOAT4; i++ )
            regs_out[i] = float( regs[i] );

        *Xfloatptr = *reinterpret_cast<float4*>( regs_out );
        *(Xfloatptr + 1) = *reinterpret_cast<float4*>( &(regs_out[4]) );
    }
}



// Converts X to Xfloat and normalizes the length
// Fully coalesced memory loads and stores
// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_X_to_Xfloat_normalize( const int16_t* X, const lentype* Xlen, float* Xfloat, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;
    const uint32_t SHORTS_PER_FLOAT4 = 8;

    const uint32_t Xstart = ROWS_PER_BLOCK * VECDIM * blockIdx.x;

    float4* Xptr = ((float4*)(X + Xstart)) + threadIdx.x;
    const float4* Xptr_end = (float4*)(X + Xstart + ROWS_PER_BLOCK * VECDIM);
    float4* Xfloatptr = ((float4*)(Xfloat + Xstart)) + 2 * threadIdx.x;

    __shared__ float s_len[ROWS_PER_BLOCK];
    s_len[threadIdx.x] =  Xlen[ROWS_PER_BLOCK * blockIdx.x + threadIdx.x];
    
    __syncthreads();

    uint32_t k = SHORTS_PER_FLOAT4*threadIdx.x;
    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xfloatptr += 2*WARP_SIZE, k+=SHORTS_PER_FLOAT4*WARP_SIZE ) {
        int16_t regs[SHORTS_PER_FLOAT4];
        float regs_out[SHORTS_PER_FLOAT4];
        *reinterpret_cast<float4*>(regs) = *Xptr;

        float normalize_factor = rsqrtf( s_len[k/VECDIM] );
        #pragma unroll 
        for( uint32_t i = 0; i < SHORTS_PER_FLOAT4; i++ )
            regs_out[i] = normalize_factor * regs[i];

        *Xfloatptr = *reinterpret_cast<float4*>( regs_out );
        *(Xfloatptr + 1) = *reinterpret_cast<float4*>( &(regs_out[4]) );
    }
}

// Converts X to Xfloat and normalizes the sign
// Fully coalesced memory loads and stores
// Call with vecs/VECS_PER_BLOCK blocks with 32 threads each
__global__ void kernel_X_to_Xfloat_negate( const int16_t* X, const iptype* Xip, float* Xfloat, const uint32_t VECDIM ) {
    const uint32_t ROWS_PER_BLOCK = 32;
    const uint32_t SHORTS_PER_FLOAT4 = 8;

    const uint32_t Xstart = ROWS_PER_BLOCK * VECDIM * blockIdx.x;

    float4* Xptr = ((float4*)(X + Xstart)) + threadIdx.x;
    const float4* Xptr_end = (float4*)(X + Xstart + ROWS_PER_BLOCK * VECDIM);
    float4* Xfloatptr = ((float4*)(Xfloat + Xstart)) + 2 * threadIdx.x;

    __shared__ float s_sign[ROWS_PER_BLOCK];
    s_sign[threadIdx.x] =  ( Xip[ROWS_PER_BLOCK * blockIdx.x + threadIdx.x] > iptype(0.)) ? 1. : -1.;
    
    __syncthreads();

    uint32_t k = SHORTS_PER_FLOAT4*threadIdx.x;
    for( ; Xptr < Xptr_end; Xptr += WARP_SIZE, Xfloatptr += 2*WARP_SIZE, k+=SHORTS_PER_FLOAT4*WARP_SIZE ) {
        int16_t regs[SHORTS_PER_FLOAT4];
        float regs_out[SHORTS_PER_FLOAT4];
        *reinterpret_cast<float4*>(regs) = *Xptr;

        float normalize_factor = s_sign[k/VECDIM];
        #pragma unroll 
        for( uint32_t i = 0; i < SHORTS_PER_FLOAT4; i++ )
            regs_out[i] = normalize_factor * regs[i];

        *Xfloatptr = *reinterpret_cast<float4*>( regs_out );
        *(Xfloatptr + 1) = *reinterpret_cast<float4*>( &(regs_out[4]) );
    }
}


// Prepare lengths and ips for wmma_sieve2
// call with vecs/128 with 32 threads each
__global__ void kernel_prepare_len_and_ips( const lentype* len_in, iptype* ips, half* len_out, float lenbound, float b_len ) {
    const int items_per_float4 = 4;
    static_assert( sizeof(lentype) == 4 );
    const uint32_t Xstart = 128 * blockIdx.x;
    
    float4* len_in_ptr = (float4*)(len_in + Xstart) + threadIdx.x;
    float2* len_out_ptr = (float2*)(len_out + Xstart) + threadIdx.x;
    float2* ips_ptr = (float2*)(ips + Xstart) + threadIdx.x;

    lenbound /= 4;
    float b_len_scaled = b_len / 4;
    float b_len_sqrt = sqrtf( b_len );

    #pragma unroll
    for( int k = 0; k < 128/items_per_float4/32; k++, len_in_ptr += WARP_SIZE, ips_ptr += WARP_SIZE, len_out_ptr += WARP_SIZE ) {
        float len_reg[items_per_float4];
        half len_out_reg[items_per_float4];
        half ips_reg[items_per_float4];

        *reinterpret_cast<float4*>(len_reg) = *len_in_ptr;
        *reinterpret_cast<float2*>(ips_reg) = *ips_ptr;

        #pragma unroll
        for( int i = 0; i < items_per_float4; i++ ) {
            len_reg[i] = 0.5 * len_reg[i] - lenbound;
            len_out_reg[i] = half(len_reg[i]);
            ips_reg[i] = half( -len_reg[i] - b_len_scaled + b_len_sqrt * fabs( float(ips_reg[i])));
        
            //if( blockIdx.x == 0 and threadIdx.x == 0 ) {
            //    printf("Prepare %f, %f, %f, %f\n", *(reinterpret_cast<float*>(len_in_ptr)+i), float(*(reinterpret_cast<half*>(ips_ptr)+i)), len_reg[i], float(ips_reg[i]));
            //}

        }



        *len_out_ptr = *reinterpret_cast<float2*>(&len_out_reg);
        *ips_ptr = *reinterpret_cast<float2*>(&ips_reg);
    }
}

template<int VECDIM, bool triple=true>
__global__
void kernel_postprocess( const float* A_yr, const float* B_yr, const float B_len, const indextype* dev_indices, const indextype* dev_nr_results, lentype* len_out) {
    constexpr int blocks = 128;
    constexpr int warps_per_block = 1;
    constexpr int elmts = VECDIM / WARP_SIZE;
    int wid = threadIdx.x / 32;

    indextype nr_results = dev_nr_results[0]; 
    const indextype max_results = VECNUM;
    nr_results = min( nr_results, max_results);

    const int2* resptr = reinterpret_cast<const int2*>(dev_indices) + blockIdx.x * warps_per_block + wid;
    const int2* resptr_end = reinterpret_cast<const int2*>(dev_indices) + nr_results;
    
    lentype* lenptr = len_out + blockIdx.x * warps_per_block + wid;

    // preload B_yr
    float B_len_reg = sqrtf(B_len);
    float B_yr_reg[elmts];
    

    if( triple ) {
        #pragma unroll
        for( int i = 0; i < elmts; i++ ) {
            B_yr_reg[i] = B_yr[WARP_SIZE*i + threadIdx.x%32];
        }
    }

    for( ; resptr < resptr_end; resptr += blocks * warps_per_block, lenptr += blocks * warps_per_block ) {
        int2 res = *resptr;
        float len = 0.;
        if( !triple or (res.x >= 0 and res.y >= 0) ) {
            // pair
            const float* A_yr_ptr_x = A_yr + VECDIM * res.x + threadIdx.x%32;
            const float* A_yr_ptr_y = A_yr + VECDIM * res.y + threadIdx.x%32;
            #pragma unroll
            for( int i = 0; i < elmts; i++ ) {
                float yrx = *(A_yr_ptr_x + i * WARP_SIZE);
                float yry = *(A_yr_ptr_y + i * WARP_SIZE);
                float yr = yrx - yry;
                len += yr * yr;
            }
        } else {
            // triple
            const float* A_yr_ptr_x = A_yr + VECDIM * (-res.x) + threadIdx.x%32;
            const float* A_yr_ptr_y = A_yr + VECDIM * (-res.y) + threadIdx.x%32;
            #pragma unroll
            for( int i = 0; i < elmts; i++ ) {
                float yrx = *(A_yr_ptr_x + i * WARP_SIZE);
                float yry = *(A_yr_ptr_y + i * WARP_SIZE);
                float yr = B_len_reg * B_yr_reg[i] - yrx - yry;
                len += yr * yr;
            }
        }

        // accumulate len
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            len += __shfl_down_sync(0xffffffff, len, offset);
        
        if( threadIdx.x%32==0 )
            *lenptr = len;
    }
}


// ------------------------ KERNELS CONTEXT CHANGES / RECOMPUTE --------------


template<int EL>
__global__
void kernel_babai( int16_t* A_x, float* A_yr, int16_t* x_extend, const float* mu, const int VECDIM, const uint32_t bucketsize) {
    constexpr int blocks = 128;
    constexpr int warps_per_block = 1;
    
    float reg_yr[EL];
    float reg_mu[EL][EL];

    // load mu ones, don't care about efficiency
    for( int i = 0; i < EL; i++ ) {
        for( int j = i; j < EL; j++ ) {
            reg_mu[i][j] = mu[i*VECDIM+j];
        }
    }

    uint32_t res = blockIdx.x * warps_per_block * WARP_SIZE + threadIdx.x;
    const uint32_t res_end = bucketsize;

    for( ; res < res_end; res += blocks * warps_per_block * WARP_SIZE ) {
        int16_t* A_x_ptr = A_x + VECDIM * res;
        int16_t* x_extend_ptr = x_extend + EL * res + EL-1;
        float* A_yr_ptr = A_yr + VECDIM * res;
        for( int i = 0; i < EL; i++, A_x_ptr++, A_yr_ptr++ ) {
            reg_yr[i] = *A_yr_ptr;
        }
        A_yr_ptr--;
        A_x_ptr--;

        for(int i = EL-1; i >= 0; i-- ) {
            // babai rounding
            int dx = -__float2int_rn( reg_yr[i] / reg_mu[i][i] );
            
            // update yr
            for(int j = i; j >= 0; j-- )
                reg_yr[j] += dx * reg_mu[j][i];
            *A_yr_ptr = reg_yr[i];
            A_yr_ptr++;

            // update global x
            *A_x_ptr += int16_t(dx);
            A_x_ptr--;
            *x_extend_ptr = int16_t(dx);
            x_extend_ptr--;  
        }
    }
}


template<int VECDIM>
__global__
void kernel_recompute_len( const float* A_yr, lentype* len_out, const uint32_t bucketsize) {
    constexpr int blocks = 128;
    constexpr int warps_per_block = 1;
    constexpr int elmts = VECDIM / WARP_SIZE;
    uint32_t wid = threadIdx.x / 32;

    uint32_t res = blockIdx.x * warps_per_block + wid;
    const uint32_t res_end = bucketsize;
    
    lentype* lenptr = len_out + blockIdx.x * warps_per_block + wid;

    for( ; res < res_end; res += blocks * warps_per_block, lenptr += blocks * warps_per_block ) {
        float len = 0.;
        const float* A_yr_ptr = A_yr + VECDIM * res + threadIdx.x%32;
        #pragma unroll
        for(uint32_t i = 0; i < elmts; i++ ) {
            float yr = *(A_yr_ptr + i * WARP_SIZE);
            len += yr * yr;
        }

        // accumulate len
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            len += __shfl_down_sync(0xffffffff, len, offset);
        
        if( threadIdx.x%32==0 )
            *lenptr = len;
    }
}

template<int VECDIM>
__global__
void kernel_check_len( const float* A_yr, lentype* len_in, const int bucketsize) {
    constexpr int blocks = 128;
    constexpr int warps_per_block = 1;
    constexpr int elmts = VECDIM / WARP_SIZE;
    int wid = threadIdx.x / 32;

    int res = blockIdx.x * warps_per_block + wid;
    const int res_end = bucketsize;
    
    lentype* lenptr = len_in + blockIdx.x * warps_per_block + wid;

    for( ; res < res_end; res += blocks * warps_per_block, lenptr += blocks * warps_per_block ) {
        float len = 0.;
        const float* A_yr_ptr = A_yr + VECDIM * res + threadIdx.x%32;
        #pragma unroll
        for( int i = 0; i < elmts; i++ ) {
            float yr = *(A_yr_ptr + i * WARP_SIZE);
            len += yr * yr;
        }

        // accumulate len
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            len += __shfl_down_sync(0xffffffff, len, offset);
        
        if( threadIdx.x%32==0 )
            if( fabsf((*lenptr) - len) > 0.01) {
                printf("Wrong length: %f, %f\n", *lenptr, len);
            }
    }
}


template<int VECDIM>
__global__
void kernel_recompute_uid( const Xtype* A_x, UidType* uid_out, const UidType* uid_coeffs, const uint32_t bucketsize) {
    constexpr int blocks = 128;
    constexpr int warps_per_block = 1;
    constexpr int elmts = VECDIM / WARP_SIZE;
    uint32_t wid = threadIdx.x / 32;

    uint32_t res = blockIdx.x * warps_per_block + wid;
    const uint32_t res_end = bucketsize;
    
    UidType* uidptr = uid_out + blockIdx.x * warps_per_block + wid;
    
    UidType reg_coeff[elmts];
    for( int i = 0; i < elmts; i++ )
        reg_coeff[i] = uid_coeffs[WARP_SIZE * i + threadIdx.x%32];



    for( ; res < res_end; res += blocks * warps_per_block, uidptr += blocks * warps_per_block ) {
        UidType uid = 0;
        const Xtype* A_x_ptr = A_x + VECDIM * res + threadIdx.x%32;
        #pragma unroll
        for( uint32_t i = 0; i < elmts; i++ ) {
            UidType x = UidType(*(A_x_ptr + i * WARP_SIZE));
            uid += x * reg_coeff[i];

        }

        // accumulate uid
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            uid += __shfl_down_sync(0xffffffff, uid, offset);

        if( threadIdx.x%32==0 )
            *uidptr = uid;
    }
}


// BDGL BUCKETING

template<int DIM, int v_packs, typename T>
inline __device__
void kernel_shuffle( T*const x, curandState*const state ) {
    // random bits to use
    int rng = curand(state);

    static_assert( DIM == 32 or DIM == 64 );
    static_assert( sizeof(T) == 4 );
   
    constexpr int h_stacks = DIM/32;
    const int rounds = 6;
    const int jj[rounds] = {21, 1, 2, 4, 8, 16 };

    #pragma unroll
    for( int r = 1; r < rounds; r++ ) {
        //share randomness
        int r_shared = rng ^ __shfl_xor_sync(0xFFFFFFFF, rng, jj[r]);
        
        #pragma unroll
        for( int h = 0; h < h_stacks; h++ ) {
            bool swap = r_shared&(1<<h);

            #pragma unroll
            for( int i = 0; i < v_packs; i++ ) {
                // conditional swap without if
                T tmp = __shfl_xor_sync(0xFFFFFFFF, x[h*v_packs+i], jj[r]);
                x[h*v_packs+i] = swap?tmp:x[h*v_packs+i];
            }
        }
        // next bits of randomness
        rng >>= h_stacks;
    }

    // local flips
    #pragma unroll
    for( int h = 0; h < h_stacks; h++ ) {
        bool flip = rng&1;
        rng >>= 1;
        #pragma unroll
        for( int i = 0; i < v_packs; i++ ) {
            x[h*v_packs+i] = flip?(-x[h*v_packs+i]):x[h*v_packs+i];
        }
    }

    // local swaps
    if( h_stacks == 2 ) {
        #pragma unroll 
        for( int i = 0; i < v_packs; i++ ) {
            if( rng&1 ) {
                T tmp = x[i];
                x[i] = x[v_packs+i];
                x[v_packs+i] = tmp;
            }
        }
    }
}

template<int DIM, int v_packs, typename T>
inline __device__
void kernel_hadamard( T* const x ) {
    const int tx = threadIdx.x;

    static_assert( DIM == 32 or DIM == 64 );
    static_assert( sizeof(T) == 4 );
    
    constexpr int h_stacks = DIM/32;

    #pragma unroll
    for( int j = 1; j < 32; j *= 2 ) {
        bool neg = (tx&j);
        
        #pragma unroll
        for( int h = 0; h < h_stacks; h++ ) {
            #pragma unroll
            for( int i = 0; i < v_packs; i++ ) {
                // conditional swap without if
                T val = __shfl_xor_sync(0xFFFFFFFF, x[h*v_packs+i], j);
                x[h*v_packs+i] = neg?(-x[h*v_packs+i]):x[h*v_packs+i]; // change sign bit (fp)
                x[h*v_packs+i] += val;
                //x[i] = val + ((tx&j)? (-x[i]) : x[i]); // check if no branch
            }
        }
    }

    // local
    if( h_stacks==2 ) {
        #pragma unroll
        for( int i = 0; i < v_packs; i++ ) {
            T tmp = x[i] - x[v_packs+i];
            x[i] += x[v_packs+i];
            x[v_packs+i] = tmp;
        }
    }
}


// results contains the best indices for vec threadIdx.x (if < 2*v_packs, otherwise empty)
// single warp
// NOTE: nr_buckets-1 should fit in a signed short
template<int VECDIM, int v_packs, int MH>
__device__ __noinline__
void kernel_bdgl_subbucketing( half2* x, curandState state, const uint32_t n, const uint32_t nr_buckets, int *results, half* results_ips ) {
	// x has shape h_packs * v_packs
    // n <= VECDIM
	static_assert(VECDIM >= 64 and VECDIM <= 128);
    constexpr int HDIM = (VECDIM>64)?64:32;
    constexpr int h_stacks = HDIM/32;
    constexpr int rh_stacks = (VECDIM-HDIM)/32;
	const int tx = threadIdx.x;
    const int remainder = n-HDIM;
	
    __shared__ int shared_res[2*v_packs*MH];
    __shared__ half shared_ips[2*v_packs*MH];

    const uint32_t batches = (nr_buckets + HDIM -1)/HDIM;

    int rng;
    
    half2 y[h_stacks*v_packs];
    
    half2 bestval[v_packs];
    for( int i = 0; i < v_packs; i++ )
        bestval[i] = __floats2half2_rn(-1., -1.); 
    int bestind[2*v_packs];
    
    for( int b = 0; b < batches; b++ ) {
        if( b%(32/rh_stacks) == 0 )
            rng = curand( &state );

        // swap remainder into hadamard part
        #pragma unroll
        for( int j = 0; j < rh_stacks; j++ ) {
			if( WARP_SIZE*j+tx < remainder and rng&1 ) {
                #pragma unroll
                for( int i = 0; i < v_packs; i++ ) {
					half2 tmp = x[(h_stacks+j)*v_packs+i];
					x[(h_stacks+j)*v_packs+i] = x[j*v_packs+i];
					x[j*v_packs+i] = tmp;
				}
			}
			rng >>= 1;
		}
            
        // shuffle HDIM part
        kernel_shuffle<HDIM, v_packs, half2>( x, &state );

        // copy
        #pragma unroll
        for( int i = 0; i < h_stacks*v_packs; i++ ) {
            y[i] = x[i];
        }

        // hadamard
        kernel_hadamard<HDIM, v_packs, half2>( y );
        
        // process results
        #pragma unroll
        for( int h = 0; h < h_stacks; h++ ) {
            #pragma unroll
            for( int i = 0; i < v_packs; i++ ) {
                if( HDIM*b+32*h+tx < nr_buckets ) {
                    half2 v = __habs2( y[v_packs * h + i] );
                    half2 rs = __hge2( v, bestval[i] );
                    //if( *reinterpret_cast<int*>(&rs) ) {
                        if( rs.x ) {
                            bestval[i].x = v.x;
                            // index based on sign
                            bestind[2*i] = (y[v_packs * h + i].x==v.x) ? HDIM*b+32*h+tx+1 : -(HDIM*b+32*h+tx+1);
                        }
                        if( rs.y ) {
                            bestval[i].y = v.y;
                            // index based on sign
                            bestind[2*i+1] = (y[v_packs * h + i].y==v.y) ? HDIM*b+32*h+tx+1 : -(HDIM*b+32*h+tx+1);
                        }
                    //}
                }
            }
        }
    }

    const half * const vals = reinterpret_cast<half*>(bestval);
    const int * const inds = reinterpret_cast<int*>(bestind);

    // gather best results
    // process vecs
    const int vecs = 2*v_packs;
    
    for( int i = 0; i < vecs; i++ ) {
        int poss = 0xFFFFFFFF;
        int k = 0;
        int res;
        int lastgoodres;
        while( poss ) {
            k = __ffs(poss)-1;
            half ipk = __shfl_sync(0xFFFFFFFF, vals[i] , k);
            res = __ballot_sync( 0xFFFFFFFF, vals[i] >= ipk ); 
            int nm = __popc(res);

            if( nm > MH ) {
                poss &= res;
                poss ^= (1 << k);
                lastgoodres = res;
            } else if( nm < MH ) {
                poss &= ~res;
            } else {
                break;
            }
        }
        
        if( __popc(res) < MH )
            res = lastgoodres;

        // non weird case
        if( __popc(res) >= MH ) {
            if( res & (1<<tx) ) {
                int j = __popc( res & ((1<<tx)-1) );
                if( j < MH ) {
                    shared_res[MH*i+j] =  inds[i];
                    shared_ips[MH*i+j] = vals[i];
                }
            }
        } else {
            // shouldn't happen?
            // strange case, just give some result
            if( tx < MH ) {
                shared_res[MH*i+tx] = inds[i];
                shared_ips[MH*i+tx] = vals[i];
            }
        }
    }
    __syncwarp();

    // no sync needed because single warp
    if( tx < vecs ) {
        for( int i = 0; i < MH; i++ ) {
            results[i] = shared_res[MH*tx+i];
            results_ips[i] = shared_ips[MH*tx+i];
        }
    }
}
template<int VECDIM, int MH>
__global__ 
void kernel_bdgl_bucketing_1block(const half* a, const int32_t n, const uint32_t batch_size, const int nr_lbuckets, uint32_t seed, uint32_t* bucket_indices, half* bucket_ips) {
    static_assert(VECDIM>=64);
    static_assert(MH <= 32);
    const int blocks = 1;
    constexpr int total_h_stacks = VECDIM/32;
    constexpr int local_h_stacks = total_h_stacks;
    constexpr int D = local_h_stacks * 32;
    
    const int vecs = 16;
    const int v_packs = vecs/2;
    const int cuda_blocks = gridDim.x;
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    
    curandState bucketState[blocks];
    for( int b = 0; b < blocks; b++ )
        curand_init( seed, tx, 50+50*b, &bucketState[b] );

    __shared__ half a_shared[vecs][VECDIM];
    const int halfs_per_float4 = 8;
    static_assert( (vecs*VECDIM)%(WARP_SIZE*halfs_per_float4) == 0 );
    const int cache_regs = vecs * VECDIM / WARP_SIZE / halfs_per_float4;
    
    float4 reg_cache[cache_regs];

    const float4* a_global = reinterpret_cast<const float4*>( a + bx * VECDIM * vecs ) + tx; 
    const float4* const a_end = reinterpret_cast<const float4*>( a + batch_size * VECDIM );

    const int A_GLOBAL_SHIFT = cuda_blocks * VECDIM * vecs / halfs_per_float4;

    // first global -> reg1
    for( int i = 0; i < cache_regs; i++ ) {
        reg_cache[i] = *(a_global + i*WARP_SIZE);
    }
    a_global += A_GLOBAL_SHIFT;

    for( int t = 0; a_global < a_end + A_GLOBAL_SHIFT; t++, a_global += A_GLOBAL_SHIFT  ) {
        
        int results[blocks][MH];
        half results_ips[blocks][MH];
        
        __syncwarp(); // theoretically not needed within a single warp

        // reg1 -> shared
        for( int i = 0; i < cache_regs; i++ )
            *reinterpret_cast<float4*>(&a_shared[(WARP_SIZE*i+tx)*8/VECDIM][(WARP_SIZE*i+tx)*8%VECDIM]) = reg_cache[i];

        __syncwarp(); // theoretically not needed within a single warp
    
        if( a_global < a_end ) {
            // global -> reg1
            for( int i = 0; i < cache_regs; i++ ) {
                reg_cache[i] = *(a_global + i*WARP_SIZE);
            }
        }

        // first block
        half2 x[local_h_stacks*v_packs];
        for( int i = 0; i < local_h_stacks; i++ ) {
            for( int j = 0; j < v_packs; j++ ) {
                if( 32*i+tx < n )
                    x[i*v_packs+j] = __halves2half2( a_shared[2*j][32*i+tx], a_shared[2*j+1][32*i+tx] );
                else
                    x[i*v_packs+j] = __floats2half2_rn(0.,0.);
            }
        }
        kernel_bdgl_subbucketing<D, v_packs, MH>( x, bucketState[0], n, nr_lbuckets, results[0], results_ips[0] );
        
        // return results
        if( tx < vecs ) {
            for( int i = 0; i < MH; i++ ) {
                bucket_indices[((t*vecs*cuda_blocks)+bx*vecs+tx)*MH+i] = abs(results[0][i])-1;
                bucket_ips[((t*vecs*cuda_blocks)+bx*vecs+tx)*MH+i] = (results[0][i]>0) ? __float2half(0.01) : __float2half(-0.01);
            }
        }
    }
}

template<int VECDIM, int MH>
__global__ 
void kernel_bdgl_bucketing_2block(const half* a, const int32_t n, const uint32_t batch_size, const int nr_lbuckets, uint32_t seed, indextype* bucket_indices, half* bucket_ips) {
    static_assert(VECDIM>=64);
    static_assert(MH <= 32);
    
    const int blocks = 2;
    constexpr int total_h_stacks = VECDIM/32;
    constexpr int local_h_stacks = (total_h_stacks+1)/2;
    constexpr int D = local_h_stacks * 32;
    const int nn[blocks] = {(n+1)/2, n/2};

    const int vecs = 8;
    const int v_packs = vecs/2;
    const int cuda_blocks = gridDim.x;
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    
    curandState bucketState[2];
    for( int b = 0; b < blocks; b++ ) {
        curand_init( seed, tx, 1+b, &bucketState[b] );
    }

    __shared__ half a_shared[vecs][VECDIM];
    const int halfs_per_float4 = 8;
    static_assert( (vecs*VECDIM)%(WARP_SIZE*halfs_per_float4) == 0 );
    const int cache_regs = vecs * VECDIM / WARP_SIZE / halfs_per_float4;
    
    float4 reg_cache[cache_regs];

    const float4* a_global = reinterpret_cast<const float4*>( a + bx * VECDIM * vecs ) + tx; 
    const float4* const a_end = reinterpret_cast<const float4*>( a + batch_size * VECDIM );

    const int A_GLOBAL_SHIFT = cuda_blocks * VECDIM * vecs / halfs_per_float4;

    // first global -> reg1
    #pragma unroll
    for( int i = 0; i < cache_regs; i++ ) {
        reg_cache[i] = *(a_global + i*WARP_SIZE);
    }
    a_global += A_GLOBAL_SHIFT;

    for( int t = 0; a_global < a_end + A_GLOBAL_SHIFT; t++, a_global += A_GLOBAL_SHIFT  ) {
        
        int results[blocks][MH];
        half results_ips[blocks][MH];

        __syncthreads(); // theoretically not needed within a single warp

        // reg1 -> shared
        #pragma unroll
        for( int i = 0; i < cache_regs; i++ )
            *reinterpret_cast<float4*>(&a_shared[(WARP_SIZE*i+tx)*8/VECDIM][(WARP_SIZE*i+tx)*8%VECDIM]) = reg_cache[i];

        __syncthreads(); // theoretically not needed within a single warp
    
        if( a_global < a_end ) {
            // global -> reg1
            for( int i = 0; i < cache_regs; i++ ) {
                reg_cache[i] = *(a_global + i*WARP_SIZE);
            }
        }

        // first block
        half2 x[local_h_stacks*v_packs];
        #pragma unroll
        for( int i = 0; i < local_h_stacks; i++ ) {
            #pragma unroll
            for( int j = 0; j < v_packs; j++ ) {
                if( 32*i+tx < nn[0] )
                    x[i*v_packs+j] = __halves2half2( a_shared[2*j][32*i+tx], a_shared[2*j+1][32*i+tx] );
                else
                    x[i*v_packs+j] = __floats2half2_rn(0.,0.);
            }
        }
        kernel_bdgl_subbucketing<D, v_packs, MH>( x, bucketState[0], nn[0], nr_lbuckets, results[0], results_ips[0] );
        
        // second block
        #pragma unroll
        for( int i = 0; i < local_h_stacks; i++ ) {
            #pragma unroll
            for( int j = 0; j < v_packs; j++ ) {
                if( 32*i+tx < nn[1] )
                    x[i*v_packs+j] = __halves2half2( a_shared[2*j][nn[0]+32*i+tx], a_shared[2*j+1][nn[0]+32*i+tx] );
                else
                    x[i*v_packs+j] = __floats2half2_rn(0.,0.);
            }
        }
        kernel_bdgl_subbucketing<D, v_packs, MH>( x, bucketState[1], nn[1], nr_lbuckets, results[1], results_ips[1] );

        // return results
        if( tx < vecs ) {
            // results_ips is already absolute value

            half best_vals[MH] = {__float2half(0.)};
            short2 best_ind[MH] = {0};
            
            #pragma unroll
            for( int i = 0; i < MH; i++ ) {
                #pragma unroll
                for( int j = 0; j < MH; j++ ) {
                    half val = results_ips[0][i] + results_ips[1][j];
                    short2 ind = make_short2(i,j);
                    #pragma unroll
                    for( int k = 0; k < MH; k++ ) {
                        if( val > best_vals[k] ) {
                            // swap val and ind
                            half tmp = best_vals[k];
                            best_vals[k] = val;
                            val = tmp;

                            short2 tmp2 = best_ind[k];
                            best_ind[k] = ind;
                            ind = tmp2;
                        }
                    }
                }
            }

            // combine results, now just take product of local solution sets 
            // (values aren't taken into account)
            int combinedResults[MH];
            #pragma unroll
            for( int i = 0; i < MH; i++ ) {
                int m0 = best_ind[i].x;
                int m1 = best_ind[i].y;


                int h0 = abs( results[0][m0] ) - 1;
                int sign0 = ( results[0][m0] > 0 ) ? 1 : -1;
                int h1 = sign0 * results[1][m1];
                h1 = 2 * (abs(h1)-1) + (h1 > 0);
                h1 = 2 * nr_lbuckets * h0 + h1;
                combinedResults[i] = sign0 * (h1+1);

            }
            
            #pragma unroll
            for( int i = 0; i < MH; i++ ) {
                bucket_indices[((t*vecs*cuda_blocks)+bx*vecs+tx)*MH+i] = abs(combinedResults[i])-1;
                bucket_ips[((t*vecs*cuda_blocks)+bx*vecs+tx)*MH+i] = (combinedResults[i]>0) ? __float2half(0.01) : __float2half(-0.01);
            }
        }
    }
}

// BDGL BUCKETING END

// cuda_tensor commit 220ca6a2022d2f3def9ab7b6dc241e64a42a4c12

// BGJ1 bucketing

// reorder kernel
template<int VECDIM>
__global__
void kernel_reorder(half* a, uint32_t bucketsize) {
    typedef rowmat_frag_traits<16,16,16,__half> frag_traits;
    typedef row_matrix<VECDIM, frag_traits, true> mat_type;
    constexpr uint32_t reg_rowblocks = 8;
    constexpr uint32_t astep = 16 * reg_rowblocks;

    uint32_t aid = blockIdx.x * 16 * reg_rowblocks;
    const uint32_t aid_end = aid + astep;

    mat_type matrix((half*)a, bucketsize); 
    matrix.prepare_fastload_partial(aid, aid_end);
}

// input a must be using half (fp16), of size 16 m x 16 k x cudablocks for integers m and k,
template<int VECDIM, int MH=4>
__global__
__launch_bounds__(256, 1)
void kernel_bucketing(half* a, half* b, const uint32_t bucketsize, uint32_t* bucket_indices, half* bucket_ips)
{
    typedef rowmat_frag_traits<16,16,16,__half> frag_traits;
    typedef row_matrix<VECDIM, frag_traits, true> mat_type;
    typedef half ctype;
   
    constexpr int warps_per_block = 8;
    constexpr int threads_per_block = warps_per_block * WARP_SIZE;
    constexpr int h_frags_per_warp = 4;
    constexpr int w_frags_per_warp = 4;

    constexpr int h_warps_per_block = 2;
    constexpr int w_warps_per_block = 4;

    constexpr int h_cache_per_warp = (h_frags_per_warp * h_warps_per_block) / warps_per_block;
    constexpr int w_cache_per_warp = (w_frags_per_warp * w_warps_per_block) / warps_per_block;
    static_assert((h_frags_per_warp*h_warps_per_block)%warps_per_block==0, "Col frags should be multiple of warps");
    static_assert((w_frags_per_warp*w_warps_per_block)%warps_per_block==0, "Row frags should be multiple of warps");

    constexpr int astep = 16 * h_frags_per_warp * h_warps_per_block;
    constexpr int bstep = 16 * w_frags_per_warp * w_warps_per_block;

    int aid = blockIdx.x * astep;
    const int aid_end = aid + astep;

    mat_type matrix((half*)a, bucketsize); // bucketsize doesn't do anything here 
    mat_type bucket((half*)b, bucketsize);

    const int tx = threadIdx.x;
    const int warpid = threadIdx.x/WARP_SIZE; 
    const int laneid = tx%WARP_SIZE;
    
    const int wx = warpid / w_warps_per_block;
    const int wy = warpid % w_warps_per_block;

    // Fragment depth of double buffering
    const int B = 2;
    static_assert(B==2);

    // Cache regs, Load from Global, write to Shared
    frag_traits::afrag_u afrags_cache[B][h_cache_per_warp];
    frag_traits::bfrag_u bfrags_cache[B][w_cache_per_warp];

    // Compute registers, Load from shared
    frag_traits::afrag_u afrags[B][h_frags_per_warp];
    frag_traits::bfrag_u bfrags[B][w_frags_per_warp];

    // Acc frags
    accfrag_traits<16,16,16,ctype>::cfrag_u cfrags[h_frags_per_warp][w_frags_per_warp]; 

    // Shared buffers for 2 rows/columns of fragments
    __shared__ float4 shared_afrags[B][h_frags_per_warp*h_warps_per_block*WARP_SIZE];
    __shared__ float4 shared_bfrags[B][w_frags_per_warp*w_warps_per_block*WARP_SIZE];

    constexpr int vecs_per_thread = 2 * h_frags_per_warp;
    uint32_t best_ind[vecs_per_thread];
    half best_ips[vecs_per_thread];
    #pragma unroll
    for( int i = 0; i < 8; i++ ) {
        best_ips[i] = half(0.);
    }

    for( ; aid < aid_end; aid += astep ) {    

        const int b_start = 0; 
        const int bid_end = bucketsize -  ((bucketsize-b_start)%bstep);
        int bid = b_start;

        // Preload G0 -> frags_cache[1]
        #pragma unroll
        for( int i = 1; i < B; i++ ) {
            #pragma unroll
            for( int j = 0; j < h_cache_per_warp; j++ ) 
               matrix.loadc_frag(&(afrags_cache[i][j]), aid + 16 * (warps_per_block*j+warpid), 0); 
            
            #pragma unroll
            for( int j = 0; j < w_cache_per_warp; j++ )
               bucket.bloadc_frag(&(bfrags_cache[i][j]), bid + 16 * (warps_per_block*j+warpid), 0); 
        }

        // frags_cache[1] -> shared_frags[0]
        #pragma unroll
        for( int j = 0; j < h_cache_per_warp; j++ )
            shared_afrags[0][threadIdx.x + j * threads_per_block] = afrags_cache[1][j].w[0];   
        #pragma unroll
        for( int j = 0; j < w_cache_per_warp; j++ ) 
            shared_bfrags[0][threadIdx.x + j * threads_per_block] = bfrags_cache[1][j].w[0];

        // Preload G1,2 -> frags_cache
        #pragma unroll
        for( int i = 0; i < B; i++ ) {
            #pragma unroll
            for( int j = 0; j < h_cache_per_warp; j++ ) 
               matrix.loadc_frag(&(afrags_cache[i][j]), aid + 16 * (warps_per_block*j+warpid), i+1); 
            
            #pragma unroll
            for( int j = 0; j < w_cache_per_warp; j++ )
               bucket.bloadc_frag(&(bfrags_cache[i][j]), bid + 16 * (warps_per_block*j+warpid), i+1); 
        }

        __syncthreads();

        // frags_cache[0] -> frags[0]
        #pragma unroll
        for( int j = 0; j < h_frags_per_warp; j++ )
            afrags[0][j].w[0] = shared_afrags[0][(wx*h_frags_per_warp+j) * WARP_SIZE + laneid];
        #pragma unroll
        for( int j = 0; j < w_frags_per_warp; j++ )
            bfrags[0][j].w[0] = shared_bfrags[0][(wy*w_frags_per_warp+j) * WARP_SIZE + laneid];
        
        for( ; bid < bid_end;) {
            // fill cfrags
            #pragma unroll
            for( int i = 0; i < h_frags_per_warp; i++ ) {
                for( int j = 0; j < w_frags_per_warp; j++ )
                    cfrags[i][j].fill( ctype(0.) );
            }
      
            #pragma unroll
            for( int r = 0; r < VECDIM/16; r+=B ) {
                
                // frags_cache -> shared_frags
                #pragma unroll
                for( int i = 0; i < B; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < h_cache_per_warp; j++ )
                        shared_afrags[i^1][threadIdx.x + j * threads_per_block] = afrags_cache[i][j].w[0];   
                    #pragma unroll
                    for( int j = 0; j < w_cache_per_warp; j++ ) 
                        shared_bfrags[i^1][threadIdx.x + j * threads_per_block] = bfrags_cache[i][j].w[0];
                }

                __syncthreads();

                // G -> frags_cache
                #pragma unroll
                for( int i = 0; i < B; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < h_cache_per_warp; j++ ) 
                       matrix.loadc_frag(&(afrags_cache[i][j]), aid + 16 * (warps_per_block*j+warpid), (r+i+3)%matrix.row_frags); 
                    if( r+i+3 == VECDIM/16 )
                        bid += bstep;
                    
                    if( bid < bid_end ) {
                        #pragma unroll
                        for( int j = 0; j < w_cache_per_warp; j++ )
                           bucket.bloadc_frag(&(bfrags_cache[i][j]), bid + 16 * (warps_per_block*j+warpid), (r+i+3)%matrix.row_frags); 
                    }
                }

                // shared_frags[1] -> frags[1]
                #pragma unroll
                for( int j = 0; j < h_frags_per_warp; j++ )
                    afrags[1][j].w[0] = shared_afrags[1][(wx*h_frags_per_warp+j) * WARP_SIZE + laneid];
                #pragma unroll
                for( int j = 0; j < w_frags_per_warp; j++ )
                    bfrags[1][j].w[0] = shared_bfrags[1][(wy*w_frags_per_warp+j) * WARP_SIZE + laneid];
            
                // Compute frags[0]
                #pragma unroll
                for( int i = 0; i < h_frags_per_warp; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < w_frags_per_warp; j++ )
                        wmma::mma_sync(cfrags[i][j].frag, afrags[0][i].frag, bfrags[0][j].frag, cfrags[i][j].frag);
                }

                // shared_frags[0] -> frags[0]
                #pragma unroll
                for( int j = 0; j < h_frags_per_warp; j++ )
                    afrags[0][j].w[0] = shared_afrags[0][(wx*h_frags_per_warp+j) * WARP_SIZE + laneid];
                #pragma unroll
                for( int j = 0; j < w_frags_per_warp; j++ )
                    bfrags[0][j].w[0] = shared_bfrags[0][(wy*w_frags_per_warp+j) * WARP_SIZE + laneid];

                // Compute frags[1]
                #pragma unroll
                for( int i = 0; i < h_frags_per_warp; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < w_frags_per_warp; j++ )
                        wmma::mma_sync(cfrags[i][j].frag, afrags[1][i].frag, bfrags[1][j].frag, cfrags[i][j].frag);
                }

                __syncthreads();
            }

            // process results
            #pragma unroll
            for( int i = 0; i < h_frags_per_warp; i++ ) {
                #pragma unroll
                for( int k = 0; k < 8; k++ ) {
                    const int p = 2*i+(k%4>1);


                    #pragma unroll
                    for( int j = 0; j < w_frags_per_warp; j++ ) { 
                        if(__habs(cfrags[i][j].frag.x[k]) > __habs(best_ips[p])) {
                            best_ips[p] = cfrags[i][j].frag.x[k];
                            best_ind[p] = (bid-bstep + 16*w_frags_per_warp*(warpid%w_warps_per_block)+16*j+ 2 * (threadIdx.x%4) + k%2 + 8*(k>=4));
                        }
                    } 
                }
            }

        }
    }

    __syncthreads();
  
    constexpr int threads = warps_per_block * WARP_SIZE;
    constexpr int vecs = h_warps_per_block * h_frags_per_warp * 16;
    constexpr int threads_per_vec = (threads*vecs_per_thread)/vecs;
        
    // reuse shared memory
    half* shared_best_ips = reinterpret_cast<half*>( &(shared_afrags[0][0]) );
    uint32_t* shared_best_ind = reinterpret_cast<uint32_t*>( &(shared_afrags[1][0]) );


    for( int i = 0; i < vecs_per_thread; i++ ) {
        shared_best_ips[ threads_per_vec * (wx*h_frags_per_warp*16 + 8 * i + (laneid/4)) + 4*wy + threadIdx.x%4 ] = best_ips[i];
        shared_best_ind[ threads_per_vec * (wx*h_frags_per_warp*16 + 8 * i + (laneid/4)) + 4*wy + threadIdx.x%4 ] = uint32_t(best_ind[i]);
    }

    __syncthreads();
    
    if( laneid < 16 ) {
 
        int mask = 0xFFFF;
        // process vecs
        for( int i = warpid; i < vecs; i+=warps_per_block ) {
            half ip = __habs(shared_best_ips[threads_per_vec*i+laneid]);
            
            int poss = 0xFFFF;
            int k = 0;
            int res;
            while( poss ) {
                k = __ffs(poss)-1;
                half ipk = __habs( shared_best_ips[threads_per_vec*i+k] );
                res = mask & __ballot_sync( mask, ip >= ipk ); 
                int nm = __popc(res);
                
                if( nm > MH ) {
                    poss &= res;
                    poss ^= (1 << k);
                } else if( nm < MH ) {
                    poss &= ~res;
                } else {
                    break;
                }
            }
            
            // non weird case
            if( __popc(res) >= MH ) {
                if( res & (1<<laneid) ) {
                    int j = __popc( res & ((1<<laneid)-1) );
                    if( j < MH ) {
                        bucket_ips[ MH * (blockIdx.x*astep + i) + j ] = shared_best_ips[threads_per_vec*i+laneid];
                        bucket_indices[ MH * (blockIdx.x*astep + i) + j ] = shared_best_ind[threads_per_vec*i+laneid];
                    }
                }
            } else {
                if( laneid < MH ) {
                    bucket_ips[ MH * (blockIdx.x*astep + i) + laneid ] = shared_best_ips[threads_per_vec*i+laneid];
                    bucket_indices[ MH * (blockIdx.x*astep + i) + laneid ] = shared_best_ind[threads_per_vec*i+laneid];
                }
            }
        }

    }

    return;
}

__shared__ uint32_t tmpwriteidx;
constexpr int max_results_per_block = 128;
__shared__ int resultbuffer[2*max_results_per_block];
__noinline__ __device__ void save_result2(unsigned int cmask, int aid, int bid, int* results)
{
    int index = atomicAdd_block(&tmpwriteidx,__popc(cmask));
    int i = __ffs( cmask );
    while( i and index < max_results_per_block) {
        resultbuffer[2*index] = ((((i-1)/8)%2)?-1:1) *  (aid + (threadIdx.x%32)/4 + 8 * (((i-1)%4) >= 2));
        resultbuffer[2*index+1] = ((((i-1)/8)%2)?-1:1) * (bid + 16*(i>16) + 2 * (threadIdx.x%4) + 8 * ((i-1)%8>=4) + ((i-1)%2));
        index ++;
        cmask ^= 1 << (i-1);
        i = __ffs(cmask);
    }
}

__shared__ uint32_t tmpwriteidx_lift;
constexpr int max_results_per_block_lift = 256;
__shared__ int2 resultbuffer_lift[max_results_per_block_lift];
__noinline__ __device__ void save_result3(unsigned int ai, int bi)
{
    int index = atomicAdd_block(&tmpwriteidx_lift,1);
    if (index < max_results_per_block_lift) {
        resultbuffer_lift[index] = make_int2(ai, bi);
    }
}

// input a must be using half (fp16), of size 16 m x 16 k x cudablocks for integers m and k,
// Assume lengths are precomputed as 1/2 * ||x||^2 - 1/4 * lenbound
// Assume ips are precomputed as 1/4 * lenbound - 1/2 * ||x||^2 - 1/4 * ||z||^2 + <x,z>
template<int VECDIM, bool TRIPLE = false>
__global__
__launch_bounds__(256, 1)
void kernel_triple_sieve(half* a, const half* len, const half* ips, const uint32_t bucketsize, uint32_t* nr_results, int* results)
{
    typedef rowmat_frag_traits<16,16,16,__half> frag_traits;
    typedef row_matrix<VECDIM, frag_traits, true> mat_type;
    typedef row_data<__half> data_type;
    typedef half ctype;
    
    constexpr int warps_per_block = 8;
    constexpr int threads_per_block = warps_per_block * WARP_SIZE;
    constexpr int h_frags_per_warp = 4;
    constexpr int w_frags_per_warp = 4;

    constexpr int h_warps_per_block = 2;
    constexpr int w_warps_per_block = 4;

    constexpr int h_cache_per_warp = (h_frags_per_warp * h_warps_per_block) / warps_per_block;
    constexpr int w_cache_per_warp = (w_frags_per_warp * w_warps_per_block) / warps_per_block;
    static_assert((h_frags_per_warp*h_warps_per_block)%warps_per_block==0, "Col frags should be multiple of warps");
    static_assert((w_frags_per_warp*w_warps_per_block)%warps_per_block==0, "Row frags should be multiple of warps");

    constexpr int astep = 16 * h_frags_per_warp * h_warps_per_block;
    constexpr int bstep = 16 * w_frags_per_warp * w_warps_per_block;

    int aid = blockIdx.x * astep;
    assert( aid < bucketsize );
    const int aid_end = aid + astep;

    mat_type matrix((half*)a, bucketsize); 
    
    data_type data_len(len, bucketsize);
    data_type data_ips(ips, bucketsize);

    const int tx = threadIdx.x;
    const int warpid = threadIdx.x/WARP_SIZE; 
    const int laneid = tx%WARP_SIZE;
    
    const int wx = warpid / w_warps_per_block;
    const int wy = warpid % w_warps_per_block;

    // Fragment depth of double buffering
    const int B = 2;
    static_assert(B==2);

    // Cache regs, Load from Global, write to Shared
    frag_traits::afrag_u afrags_cache[B][h_cache_per_warp];
    frag_traits::bfrag_u bfrags_cache[B][w_cache_per_warp];

    // Compute registers, Load from shared
    frag_traits::afrag_u afrags[B][h_frags_per_warp];
    frag_traits::bfrag_u bfrags[B][w_frags_per_warp];

    // Acc frags
    accfrag_traits<16,16,16,ctype>::cfrag_u cfrags[h_frags_per_warp][w_frags_per_warp]; 

    // Shared buffers for 2 rows/columns of fragments
    __shared__ float4 shared_afrags[B][h_frags_per_warp*h_warps_per_block*WARP_SIZE];
    __shared__ float4 shared_bfrags[B][w_frags_per_warp*w_warps_per_block*WARP_SIZE];
    __shared__ typename row_data_shared<data_type, astep, warps_per_block, true>::shared_cache_t a_shared_len;
    __shared__ typename row_data_shared<data_type, astep, warps_per_block, true>::shared_cache_t a_shared_ips;
    __shared__ typename row_data_shared<data_type, bstep, warps_per_block, false>::shared_cache_t b_shared_len;
    __shared__ typename row_data_shared<data_type, bstep, warps_per_block, false>::shared_cache_t b_shared_ips;

    row_data_shared<data_type, astep, warps_per_block, true> a_data_cache;
    row_data_shared<data_type, bstep, warps_per_block, false> b_data_cache;

    if( tx == 0 )
        tmpwriteidx = 0;
    for( ; aid < aid_end; aid += astep ) {    

        const int b_start = 0; //aid+astep; 
        const int bid_end = aid - ((aid-b_start)%bstep);//bucketsize -  ((bucketsize-b_start)%bstep);
        if( b_start >= bid_end )
            continue;
        int bid = b_start;

        a_data_cache.global_to_shared_rows( data_len, a_shared_len, aid );    
        a_data_cache.global_to_shared_rows( data_ips, a_shared_ips, aid );
        b_data_cache.global_to_shared_rows( data_len, b_shared_len, bid );    
        b_data_cache.global_to_shared_rows( data_ips, b_shared_ips, bid );

        // Preload G0 -> frags_cache[1]
        #pragma unroll
        for( int i = 1; i < B; i++ ) {
            #pragma unroll
            for( int j = 0; j < h_cache_per_warp; j++ ) 
               matrix.loadc_frag(&(afrags_cache[i][j]), aid + 16 * (warps_per_block*j+warpid), 0); 
            
            #pragma unroll
            for( int j = 0; j < w_cache_per_warp; j++ )
               matrix.bloadc_frag(&(bfrags_cache[i][j]), bid + 16 * (warps_per_block*j+warpid), 0); 
        }
        
        // frags_cache[1] -> shared_frags[0]
        #pragma unroll
        for( int j = 0; j < h_cache_per_warp; j++ )
            shared_afrags[0][threadIdx.x + j * threads_per_block] = afrags_cache[1][j].w[0];   
        #pragma unroll
        for( int j = 0; j < w_cache_per_warp; j++ ) 
            shared_bfrags[0][threadIdx.x + j * threads_per_block] = bfrags_cache[1][j].w[0];

        // Preload G1,2 -> frags_cache
        #pragma unroll
        for( int i = 0; i < B; i++ ) {
            #pragma unroll
            for( int j = 0; j < h_cache_per_warp; j++ ) 
               matrix.loadc_frag(&(afrags_cache[i][j]), aid + 16 * (warps_per_block*j+warpid), i+1); 
            
            #pragma unroll
            for( int j = 0; j < w_cache_per_warp; j++ )
               matrix.bloadc_frag(&(bfrags_cache[i][j]), bid + 16 * (warps_per_block*j+warpid), i+1); 
        }

        __syncthreads();

        // frags_cache[0] -> frags[0]
        #pragma unroll
        for( int j = 0; j < h_frags_per_warp; j++ )
            afrags[0][j].w[0] = shared_afrags[0][(wx*h_frags_per_warp+j) * WARP_SIZE + laneid];
        #pragma unroll
        for( int j = 0; j < w_frags_per_warp; j++ )
            bfrags[0][j].w[0] = shared_bfrags[0][(wy*w_frags_per_warp+j) * WARP_SIZE + laneid];
        
        for( ; bid < bid_end;) {
            // fill cfrags
            #pragma unroll
            for( int i = 0; i < h_frags_per_warp; i++ ) {
                for( int j = 0; j < w_frags_per_warp; j++ )
                    cfrags[i][j].fill( ctype(0.) );
            }
      
            #pragma unroll
            for( int r = 0; r < matrix.row_frags; r+=B ) {
                
                // frags_cache -> shared_frags
                #pragma unroll
                for( int i = 0; i < B; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < h_cache_per_warp; j++ )
                        shared_afrags[i^1][threadIdx.x + j * threads_per_block] = afrags_cache[i][j].w[0];   
                    #pragma unroll
                    for( int j = 0; j < w_cache_per_warp; j++ ) 
                        shared_bfrags[i^1][threadIdx.x + j * threads_per_block] = bfrags_cache[i][j].w[0];
                }

                __syncthreads();

                if( r == 0 ) {
                    b_data_cache.global_to_shared_rows( data_len, b_shared_len, bid );    
                    b_data_cache.global_to_shared_rows( data_ips, b_shared_ips, bid );
                }

                // G -> frags_cache
                #pragma unroll
                for( int i = 0; i < B; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < h_cache_per_warp; j++ ) 
                       matrix.loadc_frag(&(afrags_cache[i][j]), aid + 16 * (warps_per_block*j+warpid), (r+i+3)%matrix.row_frags); 
                    
                    if( r+i+3 == matrix.row_frags )
                        bid += bstep;
                    
                    if( bid < bid_end ) {
                        #pragma unroll
                        for( int j = 0; j < w_cache_per_warp; j++ )
                           matrix.bloadc_frag(&(bfrags_cache[i][j]), bid + 16 * (warps_per_block*j+warpid), (r+i+3)%matrix.row_frags); 
                    }
                }

                // shared_frags[1] -> frags[1]
                #pragma unroll
                for( int j = 0; j < h_frags_per_warp; j++ )
                    afrags[1][j].w[0] = shared_afrags[1][(wx*h_frags_per_warp+j) * WARP_SIZE + laneid];
                #pragma unroll
                for( int j = 0; j < w_frags_per_warp; j++ )
                    bfrags[1][j].w[0] = shared_bfrags[1][(wy*w_frags_per_warp+j) * WARP_SIZE + laneid];
            
                // Compute frags[0]
                #pragma unroll
                for( int i = 0; i < h_frags_per_warp; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < w_frags_per_warp; j++ )
                        wmma::mma_sync(cfrags[i][j].frag, afrags[0][i].frag, bfrags[0][j].frag, cfrags[i][j].frag);
                }

                // shared_frags[0] -> frags[0]
                #pragma unroll
                for( int j = 0; j < h_frags_per_warp; j++ )
                    afrags[0][j].w[0] = shared_afrags[0][(wx*h_frags_per_warp+j) * WARP_SIZE + laneid];
                #pragma unroll
                for( int j = 0; j < w_frags_per_warp; j++ )
                    bfrags[0][j].w[0] = shared_bfrags[0][(wy*w_frags_per_warp+j) * WARP_SIZE + laneid];

                // Compute frags[1]
                #pragma unroll
                for( int i = 0; i < h_frags_per_warp; i++ ) {
                    #pragma unroll
                    for( int j = 0; j < w_frags_per_warp; j++ )
                        wmma::mma_sync(cfrags[i][j].frag, afrags[1][i].frag, bfrags[1][j].frag, cfrags[i][j].frag);
                }

                __syncthreads();
            }
                
            // load a data 
            half2 b_len[w_frags_per_warp][2];
            half2 b_ips[w_frags_per_warp][2];
            for( int j = 0; j < w_frags_per_warp; j++ ) {
                for( int shift = 0; shift < 2; shift++ ) {
                    b_len[j][shift] = b_data_cache.bload_fragment_data( b_shared_len, w_frags_per_warp * wy + j, shift);
                    b_ips[j][shift] = b_data_cache.bload_fragment_data( b_shared_ips, w_frags_per_warp * wy + j, shift);
                }
            }

            // process results
            #pragma unroll
            for( int i = 0; i < h_frags_per_warp; i++ ) {
                // load a data
                half2 a_len = a_data_cache.load_fragment_data( a_shared_len, h_frags_per_warp * wx + i );
                half2 a_ips = a_data_cache.load_fragment_data( a_shared_ips, h_frags_per_warp * wx + i );

                int cmask = 0;
                #pragma unroll
                for( int j = 0; j < w_frags_per_warp; j++ ) {

                    half2 A[4];
                    A[0] = __hadd2( half2(a_len.x, a_len.x), b_len[j][0] );
                    A[1] = __hadd2( half2(a_len.y, a_len.y), b_len[j][0] );
                    A[2] = __hadd2( half2(a_len.x, a_len.x), b_len[j][1] );
                    A[3] = __hadd2( half2(a_len.y, a_len.y), b_len[j][1] );
                    #pragma unroll
                    for( int k = 0; k < 4; k++ ) {
                        if(cfrags[i][j].frag.x[2*k] > ctype(A[k].x)) {
                            cmask |= (unsigned int)(1) << (16*(j%2)+2*k);
                        }
                        if(cfrags[i][j].frag.x[2*k+1] > ctype(A[k].y)) {
                            cmask |= (unsigned int)(1) << (16*(j%2)+2*k+1);
                        }
                    }
                    
                    if( TRIPLE ) {
                        half2 B[4];
                        B[0] = __hadd2( half2(a_ips.x, a_ips.x), b_ips[j][0] );
                        B[1] = __hadd2( half2(a_ips.y, a_ips.y), b_ips[j][0] );
                        B[2] = __hadd2( half2(a_ips.x, a_ips.x), b_ips[j][1] );
                        B[3] = __hadd2( half2(a_ips.y, a_ips.y), b_ips[j][1] );
                        #pragma unroll
                        for( int k = 0; k < 4; k++ ) {    
                            if(cfrags[i][j].frag.x[2*k] < ctype(B[k].x)) {
                                cmask |= (unsigned int)(1) << (16*(j%2)+8+2*k);
                            }
                            if(cfrags[i][j].frag.x[2*k+1] < ctype(B[k].y)) {
                                cmask |= (unsigned int)(1) << (16*(j%2)+8+2*k+1);
                            }
                        }

                    }

                    if( j%2 == 1 and cmask ) {
                        save_result2(cmask, aid + 16 * (warpid/w_warps_per_block) * h_frags_per_warp + 16 * i,  bid - bstep + 16 * w_frags_per_warp * (warpid%w_warps_per_block) + 16*(j-1), results);
                        cmask = 0;
                    }
                }
            }

        }
    }
   
    __syncthreads();
 
    __shared__ uint32_t globalIndex;
    uint32_t nr = min( tmpwriteidx, uint32_t(max_results_per_block) );
    /*if( tmpwriteidx > nr and tx == 0 ) {
        printf("Result flood %d, %d\n", blockIdx.x, tmpwriteidx);
    }*/
    if( tx == 0 ) {
        globalIndex = atomicAdd( nr_results, nr );
    }

    __syncthreads();

    const uint32_t max_results = VECNUM;
    if( globalIndex < max_results ) {
        nr = min( nr, max_results - globalIndex);

        for( int i = tx; i < nr; i += warps_per_block * WARP_SIZE ) {
            *reinterpret_cast<int2*>(&results[2*(globalIndex+i)]) = *reinterpret_cast<int2*>(&resultbuffer[2*i]);
        }

    }
    return;
}

// A has type char4 = 4*int8
// 16 | n_
// Start with N / BLH blocks and 128 threads
template<uint32_t n_>
__global__ void kernel_dualhash( const uint* A, const uint lenbound, indextype* lift_indices, indextype* nr_results ) { 
    
    const uint32_t BL = 4;
    const uint32_t BLH = 128;
    const uint32_t BLW = 64;
    const uint32_t TH = 8;
    const uint32_t TW = 8;
    const uint n = n_/4;
    __shared__ uint4 smem_a[n][BLH/4];
    __shared__ uint4 smem_b[n][BLW/4];
  

    const uint bx = blockIdx.x; //indicates which row batch of size BLH
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint tid = blockDim.x*ty + tx;
    const uint ti = tid % 32; // consecutive numbering inside warp
    const uint wid = tid/32;
    const uint warps_per_block = 4;
    const uint threads_per_block = warps_per_block * 32;

    uint4 const * aGlobal = reinterpret_cast<const uint4*>(&A[bx * n * BLH]);
    uint4 const * const aGlobalEnd = reinterpret_cast<const uint4*>(A + (gridDim.x*BLH*n));
    
    if( tx == 0 and ty==0 )
        tmpwriteidx_lift = 0;
   
            
    // fill smem_a once
    if( wid < BLH/32 ) {
        for( uint i = 0; i < n/4; i++) {
            //if(  size_t(&(reinterpret_cast<uint4*>(smem_a[4*i+ti/8])[8*wid+(ti%8)]))%16 != 0 ) 
            //    printf("%p, %d, %d, %d, %p\n", &(reinterpret_cast<uint4*>(smem_a[ti/8])[8*wid+(ti%8)]), wid, ti, i, smem_a);  

            smem_a[4*i+ti/8][8*wid+(ti%8)] = *(aGlobal + 8*n*wid + 32*i + ti);
        }
    }

    // skip diagonal
    aGlobal += n * BLH / 4;

    // relative load
    aGlobal += tid;

    const uint total_parts = n*BLW/4;
    const uint LPT = (total_parts+threads_per_block-1)/threads_per_block;
    uint4 Gtmp[LPT];
    
    // coal. read b from global
    #pragma unroll
    for( uint i = 0; i < LPT; i++ ) {
        if( aGlobal < aGlobalEnd )
            Gtmp[i] = *aGlobal;
        aGlobal += threads_per_block;
    }
    // compensate for n=4, n=12
    aGlobal -= LPT * threads_per_block - total_parts;
    
    const uint its = BLH/BLW * (gridDim.x-bx-1);
    for(uint p = 0; p < its; p++ ) {
        __syncthreads();
        
        // smem <- regG
        #pragma unroll
        for( uint i = 0; i < LPT; i++ ) {
            uint tmp = threads_per_block*i+tid;
            if( tmp < total_parts )
                smem_b[ (tmp/8)%n ][ 8 * (tmp/(8*n)) + (tmp%8)] = Gtmp[i];
        }

        __syncthreads();

        // regG <- Global
        #pragma unroll
        for( uint i = 0; i < LPT; i++ ) {
            if( aGlobal < aGlobalEnd )
                Gtmp[i] = *aGlobal;
            aGlobal += threads_per_block;
        }
        // compensate for n=4, n=12
        aGlobal -= LPT * threads_per_block - total_parts;

        int reg_c[TH][TW];
        #pragma unroll
        for( uint i = 0; i < TH; i++ ) { 
            #pragma unroll
            for( uint j = 0; j < TW; j++ )
                reg_c[i][j] = 0;
        }

        for( uint k = 0; k < n; k++ ) {
            uint reg_a[TH];
            uint reg_b[TW];

            // coal. read from smem to register
            #pragma unroll
            for( uint j = 0; j < TH/4; j++ )
                reinterpret_cast<uint4*>(reg_a)[j] = smem_a[k][ty*TH/4+j];
            // coal. read from smem to register
            #pragma unroll
            for( uint j = 0; j < TW/4; j++ )
                reinterpret_cast<uint4*>(reg_b)[j] = smem_b[k][tx*TW/4+j];

            #pragma unroll
            for( uint i = 0; i < TH; i++ ) {
                #pragma unroll
                for( uint j = 0; j < TW; j++ ) {
                    // ignore carry bits
                    uint tmp1 = reg_a[i] - reg_b[j];
                    //uint tmp1 =__vsub4( reg_a[i], reg_b[j] );
                    int tmp2 = *reinterpret_cast<int*>(&tmp1); 
                    //char4 tmp2 = static_cast<char4>(tmp1);  

                    // overflow should handle modulo
                    // char4 inter. gives result in interval [-128, 127].
                    reg_c[i][j] = __dp4a( tmp2, tmp2, reg_c[i][j] );
                }
            }
        }
        
        #pragma unroll
        for( uint i = 0; i < TH; i++ ) {
            #pragma unroll
            for( uint j = 0; j < TW; j++ ) {
                if( reg_c[i][j] < lenbound ) {    
                    save_result3(BLH*bx+TH*ty+i, BLH*(bx+1)+ p*BLW+TW*tx+j); 
                }
            }
        }

    }
   
    __syncthreads();

    __shared__ uint32_t globalIndex;
    const uint32_t max_results = VECNUM;
    uint32_t nr = min( tmpwriteidx_lift, uint32_t(max_results_per_block_lift) );
    if( tx == 0 and ty == 0 ) {
        globalIndex = atomicAdd( nr_results, nr );
    }

    __syncthreads();

    if( globalIndex < max_results ) {
        nr = min( nr, max_results - globalIndex);

        for( int i = blockDim.x*ty + tx; i < nr; i += warps_per_block * WARP_SIZE ) {
            *reinterpret_cast<int2*>(&lift_indices[2*(globalIndex+i)]) = resultbuffer_lift[i];
        }
    }

}

template<typename T>
__global__
void kernel_print( T* vec, int elements ) {
    if( threadIdx.x == 0 ) {
        printf("Printing vec: ");
        for( int i = 0; i < elements; i++ )
            printf("%f, ", double(vec[i]));
        printf("\n");

    }
}

inline void convert8floats2half( const float* floats, half* halfs ) { 
    __m256 float_vector = _mm256_loadu_ps(floats);
    __m128i half_vector = _mm256_cvtps_ph(float_vector, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    _mm_store_si128((__m128i*)halfs, half_vector);
}

GPUStreamGeneral::GPUStreamGeneral(const int _device, const size_t _n, const std::string _gpu_bucketer, const bool _gpu_triple, const size_t _multi_bucket, const size_t _max_nr_buckets, const bool _global, std::vector<Entry> & _db, std::vector<CompressedEntry> &_cdb, const size_t _dual_hash_vecs) : device(_device), n(_n), bucketer(_gpu_bucketer), triple(_gpu_triple), multi_bucket(_multi_bucket), max_nr_buckets(_max_nr_buckets), global(_global), db(_db.data()), cdb(_cdb.data()), VECDIM( n_to_vecdim(_n)), lift(_dual_hash_vecs>0), dual_hash_vecs(_dual_hash_vecs) {
            CUDA_CHECK( cudaSetDevice( _device ) );
        }

void GPUStreamGeneral::malloc( global_dev_ptrs& dev_ptrs ) {

            // Set device just to be sure
            CUDA_CHECK( cudaSetDevice( device ) );

            // Create events used for syncing
            CUDA_CHECK( cudaEventCreateWithFlags( &H2D, cudaEventDisableTiming) ); 
            CUDA_CHECK( cudaEventCreateWithFlags( &D2H, cudaEventDisableTiming) );
          
            // create events for benchmarking
            if( benchmark ) {
                for( int i = 0; i < 2; i++ ) {
                    CUDA_CHECK( cudaEventCreate( &start[i] ));
                    CUDA_CHECK( cudaEventCreate( &stop[i] ));
                }
            }

            // Cuda Stream
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            cublasCreate(&handle);
            cublasSetStream(handle, stream); 
            //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH );

            // Malloc pinned memory
            size_t Xsize = size_t(VECDIM)*size_t(VECNUM);
            size_t lensize = VECNUM;
            size_t max_results = std::max(multi_bucket, size_t(2)) * VECNUM;
            size_t max_lift_results = 2*VECNUM;


            CUDA_CHECK( cudaMallocHost(&host_X, Xsize*sizeof(Xtype)) );
            CUDA_CHECK( cudaMallocHost(&host_X_extend, size_t(VECNUM*MAX_EXTEND)*sizeof(Xtype)) );
            CUDA_CHECK( cudaMallocHost(&host_len_in, lensize*sizeof(lentype)) );
            CUDA_CHECK( cudaMallocHost(&host_len_out, lensize*sizeof(lentype)) );
            CUDA_CHECK( cudaMallocHost(&host_lift_len_out, lensize*sizeof(lentype)) );
            CUDA_CHECK( cudaMallocHost(&host_ips, max_results * sizeof(iptype)) );
            CUDA_CHECK( cudaMallocHost(&host_indices, max_results * sizeof(indextype)) );
            CUDA_CHECK( cudaMallocHost(&host_lift_indices, max_lift_results * sizeof(indextype)) );
            CUDA_CHECK( cudaMallocHost(&host_nr_results, 2 * sizeof(indextype)) );

            host_nr_results[0] = host_nr_results[1] = indextype(0);

            // Malloc device memory
            size_t Bsize = size_t(VECDIM) * max_nr_buckets;
            size_t DH_size = size_t(VECNUM) * dual_hash_vecs; 

            CUDA_CHECK( cudaMalloc(&dev_X, Xsize*sizeof(Xtype)) );
            CUDA_CHECK( cudaMalloc(&dev_X_half, Xsize*sizeof(half)) );
            CUDA_CHECK( cudaMalloc(&dev_X_float, Xsize*sizeof(float)) );
            CUDA_CHECK( cudaMalloc(&dev_YR_half, Xsize*sizeof(half)) );
            CUDA_CHECK( cudaMalloc(&dev_YR_float, Xsize*sizeof(float)) );
            CUDA_CHECK( cudaMalloc(&dev_DH, DH_size*sizeof(uint8_t)) );
            CUDA_CHECK( cudaMalloc(&dev_X_extend, size_t(VECNUM*MAX_EXTEND)*sizeof(Xtype)) );
            CUDA_CHECK( cudaMalloc(&dev_len_in, lensize*sizeof(lentype)) );
            CUDA_CHECK( cudaMalloc(&dev_len_out, lensize*sizeof(lentype)) );
            CUDA_CHECK( cudaMalloc(&dev_lift_len_out, lensize*sizeof(lentype)) );
            CUDA_CHECK( cudaMalloc(&dev_len_half, lensize*sizeof(half)) );
            CUDA_CHECK( cudaMalloc(&dev_ips, max_results * sizeof(iptype)) );
            CUDA_CHECK( cudaMalloc(&dev_indices, max_results * sizeof(indextype)) );
            CUDA_CHECK( cudaMalloc(&dev_lift_indices, max_lift_results * sizeof(indextype)) );
            CUDA_CHECK( cudaMalloc(&dev_nr_results, 2 * sizeof(indextype)) );



            if( global ) {
                CUDA_CHECK( cudaMalloc(&dev_B, Bsize*sizeof(half)) );
                CUDA_CHECK( cudaMalloc(&dev_B_float, Bsize*sizeof(float)) );

                // compute mu size
                size_t mu_L_size = MAX_LIFT * VECDIM;
                size_t mu_R_size = VECDIM * VECDIM;
                size_t uid_size = VECDIM;
                size_t dual_hash_size = dual_hash_vecs * MAX_LIFT;
                CUDA_CHECK( cudaMalloc(&dev_mu_L, mu_L_size * sizeof(float)) );
                CUDA_CHECK( cudaMalloc(&dev_mu_R, mu_R_size * sizeof(float)) );
                CUDA_CHECK( cudaMalloc(&dev_q_transform, mu_R_size * sizeof(float)) );
                CUDA_CHECK( cudaMalloc(&dev_dual_hash, dual_hash_size * sizeof(float)) ); 
                CUDA_CHECK( cudaMalloc(&dev_uid_coeffs, uid_size * sizeof(UidType)) );

                dev_ptrs = { dev_mu_L, dev_mu_R, dev_uid_coeffs, dev_dual_hash, dev_B, dev_B_float, dev_q_transform };
            } else {
                dev_mu_L = dev_ptrs.dev_mu_L;
                dev_mu_R = dev_ptrs.dev_mu_R;
                dev_dual_hash = dev_ptrs.dev_dual_hash;
                dev_uid_coeffs = dev_ptrs.dev_uid_coeffs;
                dev_B = dev_ptrs.dev_B;
                dev_B_float = dev_ptrs.dev_B_float;
                dev_q_transform = dev_ptrs.dev_q_transform;
            } 

            
            //std::cerr << "Malloc, device: " << device << ", global: " << global << " " << dev_B << std::endl << " " << dev_ptrs.dev_B << std::endl;
}

void GPUStreamGeneral::free() {
            // Destory events and stream
            CUDA_CHECK( cudaEventDestroy( H2D ) );
            CUDA_CHECK( cudaEventDestroy( D2H ) );
            CUDA_CHECK( cudaStreamDestroy( stream ));
            cublasDestroy(handle);

            // Free pinned memory
            CUDA_CHECK( cudaFreeHost( host_X ) );
            CUDA_CHECK( cudaFreeHost( host_X_extend ) );
            CUDA_CHECK( cudaFreeHost( host_len_in ) );
            CUDA_CHECK( cudaFreeHost( host_len_out ) );
            CUDA_CHECK( cudaFreeHost( host_lift_len_out ) );
            CUDA_CHECK( cudaFreeHost( host_ips ) );
            CUDA_CHECK( cudaFreeHost( host_indices ) );
            CUDA_CHECK( cudaFreeHost( host_lift_indices ) );
            CUDA_CHECK( cudaFreeHost( host_nr_results ) );

            host_X = nullptr;
            host_X_extend = nullptr;
            host_len_in = nullptr;
            host_len_out = nullptr;
            host_lift_len_out = nullptr;
            host_ips = nullptr;
            host_indices = nullptr;
            host_lift_indices = nullptr;
            host_nr_results = nullptr;

            CUDA_CHECK( cudaFree( dev_X ) );
            CUDA_CHECK( cudaFree( dev_X_half ) );
            CUDA_CHECK( cudaFree( dev_X_float ) );
            CUDA_CHECK( cudaFree( dev_YR_half ) );
            CUDA_CHECK( cudaFree( dev_YR_float ) );
            CUDA_CHECK( cudaFree( dev_DH ) );
            CUDA_CHECK( cudaFree( dev_X_extend ) );
            CUDA_CHECK( cudaFree( dev_len_in ) );
            CUDA_CHECK( cudaFree( dev_len_out ) );
            CUDA_CHECK( cudaFree( dev_lift_len_out ) );
            CUDA_CHECK( cudaFree( dev_len_half ) );
            CUDA_CHECK( cudaFree( dev_ips ) );
            CUDA_CHECK( cudaFree( dev_indices ) );
            CUDA_CHECK( cudaFree( dev_lift_indices ) );
            CUDA_CHECK( cudaFree( dev_nr_results ) );
            
            dev_X = nullptr;
            dev_X_half = nullptr;
            dev_YR_half = nullptr;
            dev_DH = nullptr;
            dev_X_extend = nullptr;
            dev_len_in = nullptr;
            dev_len_out = nullptr;
            dev_lift_len_out = nullptr;
            dev_ips = nullptr;
            dev_indices = nullptr;
            dev_lift_indices = nullptr;
            dev_nr_results = nullptr;

            if( global ) {
                CUDA_CHECK( cudaFree( dev_B ) );
                CUDA_CHECK( cudaFree( dev_B_float ) );
                CUDA_CHECK( cudaFree( dev_mu_L ) );
                CUDA_CHECK( cudaFree( dev_mu_R ) );
                CUDA_CHECK( cudaFree( dev_uid_coeffs ) );
                CUDA_CHECK( cudaFree( dev_dual_hash ) );
                CUDA_CHECK( cudaFree( dev_q_transform ) );
            }

            dev_B = nullptr;
            dev_B_float = nullptr;
            dev_mu_L = nullptr;
            dev_mu_R = nullptr;
            dev_uid_coeffs = nullptr;
            dev_dual_hash = nullptr;
            dev_q_transform = nullptr;
}
 

void GPUStreamGeneral::bind(std::vector<Entry> & _db, std::vector<CompressedEntry> & _cdb) {
    CUDA_CHECK( cudaSetDevice( device ) );
    db = _db.data();
    cdb = _cdb.data();
    reset_results();
}

void GPUStreamGeneral::reset_results() {
    cdb_range = { 0,0 };
    cdb_range_prev = { 0,0 };
}


// sets general global data that is used by all kernels on a device
// -- mu
// -- uid
void GPUStreamGeneral::global_init(const std::vector<std::vector<float>> &mu_R, const std::vector<UidType> &uid_coeffs ) {
    curr_bucket = nullptr;
    prev_bucket = nullptr;

    cdb_range = { 0,0 };
    cdb_range_prev = { 0,0 };

    if( !global ) 
        return;

    // collect correct data
    // reuse host_ip for this
    size_t mu_R_size = VECDIM * VECDIM;
    size_t uid_size = VECDIM;
    
    assert( mu_R_size + uid_size <= VECNUM );

    std::memset( host_len_in, 0, (mu_R_size+2*uid_size)*sizeof(float));
    
    // mu_R
    n = mu_R.size();
    for( size_t i = 0; i < n; i++ ) {
        for( size_t j = 0; j < n; j++ ) {
            host_len_in[i * VECDIM+j] = mu_R[i][j];
        }
    }  
    
    // uid_coeffs
    std::copy( uid_coeffs.cbegin(), uid_coeffs.cbegin()+n, reinterpret_cast<UidType*>(host_len_in + mu_R_size) );
    
    CUDA_CHECK( cudaMemcpyAsync(dev_mu_R, host_len_in, mu_R_size * sizeof(float), cudaMemcpyHostToDevice, stream) ); 
    CUDA_CHECK( cudaMemcpyAsync(dev_uid_coeffs, host_len_in+mu_R_size, uid_size * sizeof(UidType), cudaMemcpyHostToDevice, stream) );
    // watch out for different type size of uid

    // Wait for all data to arrive 
    CUDA_CHECK( cudaDeviceSynchronize() );
}

void GPUStreamGeneral::dh_init(const std::vector<std::vector<float>> &mu_L, const std::vector<std::vector<float>> &dh_vecs ) {
    
    if( !lift or !global ) 
        return;

    // collect correct data
    // reuse host_ip for this
    size_t mu_L_size = MAX_LIFT * VECDIM;
    size_t dual_hash_size = MAX_LIFT * dual_hash_vecs;
    
    assert( mu_L_size + dual_hash_size <= VECNUM );

    // Convert mu's to half
    std::memset( host_len_in, 0, (mu_L_size+dual_hash_size)*sizeof(float));
    
    // mu_L
    for( size_t i = 0; i < mu_L.size(); i++ ) {
        for( size_t j = 0; j < mu_L[0].size(); j++ ) {
            host_len_in[i * VECDIM+j] = mu_L[i][j];
        }
    }

    // dual hash
    for( size_t i = 0; i < dh_vecs.size(); i++ ) {
        for( size_t j = 0; j < dh_vecs[0].size(); j++ ) {
            host_len_in[mu_L_size + i * MAX_LIFT + j] = dh_vecs[i][j];
        }
    }

    CUDA_CHECK( cudaMemcpyAsync(dev_mu_L, host_len_in, mu_L_size * sizeof(float), cudaMemcpyHostToDevice, stream) ); 
    CUDA_CHECK( cudaMemcpyAsync(dev_dual_hash, host_len_in+mu_L_size, dual_hash_size * sizeof(float), cudaMemcpyHostToDevice, stream) );
 
    // Wait for all data to arrive 
    CUDA_CHECK( cudaDeviceSynchronize() );
}
 
// ------------------------ HELPER ---------------------- //

template<bool send_length, bool cdb_index>
void GPUStreamGeneral::send_X_range( const size_t cdb_start, const size_t cdb_end ) {
    size_t bucketsize = cdb_end - cdb_start;
    assert( bucketsize % 16 == 0 );

    // make sure that pinned memory is free again
    CUDA_CHECK( cudaEventSynchronize( H2D ) );

    for( uint32_t i = 0; i < bucketsize; ++i ) {
        indextype dbi = cdb_index ? cdb[cdb_start+i].i : cdb_start+i;
        if( send_length )
            host_len_in[i] = lentype( db[dbi].len );
        
        memcpy( &host_X[VECDIM*i], &db[dbi].x[0], VECDIM * sizeof(Xtype) );
    }
    
    CUDA_CHECK( cudaMemcpyAsync(dev_X, host_X, bucketsize * VECDIM * sizeof(Xtype), cudaMemcpyHostToDevice, stream) );
    
    if( send_length )
        CUDA_CHECK( cudaMemcpyAsync(dev_len_in, host_len_in, bucketsize * sizeof(lentype), cudaMemcpyHostToDevice, stream) );

    CUDA_CHECK( cudaEventRecord( H2D, stream ) );
}

template <bool send_length, bool cdb_index, bool event_record, class Iterator>
void GPUStreamGeneral::send_X_range( const Iterator first, const Iterator last ) {
    size_t bucketsize = std::distance( first, last );
    assert( bucketsize % 16 == 0 );

    // make sure that pinned memory is free again
    CUDA_CHECK( cudaEventSynchronize( H2D ) );
    
    Iterator it = first;
    for( uint32_t i = 0; i < bucketsize; ++i, it++ ) {
        indextype dbi = (cdb_index) ? cdb[*it].i : *it;
        if( send_length )
            host_len_in[i] = lentype( db[dbi].len );
        
        memcpy( &host_X[VECDIM*i], &db[dbi].x[0], VECDIM * sizeof(Xtype) );
    }
    
    CUDA_CHECK( cudaMemcpyAsync(dev_X, host_X, bucketsize * VECDIM * sizeof(Xtype), cudaMemcpyHostToDevice, stream) );
    
    if( send_length )
        CUDA_CHECK( cudaMemcpyAsync(dev_len_in, host_len_in, bucketsize * sizeof(lentype), cudaMemcpyHostToDevice, stream) );

    if( event_record )
        CUDA_CHECK( cudaEventRecord( H2D, stream ) );
}

inline void GPUStreamGeneral::X_to_Xhalf( const size_t bucketsize ) {
    // kernel threads/blocks
    
    const size_t blocks = bucketsize / 32;
    const size_t threads = 32;
    kernel_X_to_Xhalf<<<blocks, threads, 0, stream>>>( dev_X, dev_X_half, VECDIM );
}

inline void GPUStreamGeneral::X_to_Xhalf_negate( const size_t bucketsize ) {
    // kernel threads/blocks
    
    const size_t blocks = bucketsize / 32;
    const size_t threads = 32;
    kernel_X_to_Xhalf_negate<<<blocks, threads, 0, stream>>>( dev_X, dev_ips, dev_X_half, VECDIM );
}

inline void GPUStreamGeneral::X_to_Xhalf_normalize( const size_t bucketsize ) {
    // kernel threads/blocks
    
    const size_t blocks = bucketsize / 32;
    const size_t threads = 32;
    kernel_X_to_Xhalf_normalize<<<blocks, threads, 0, stream>>>( dev_X, dev_len_in, dev_X_half, VECDIM );
}

inline void GPUStreamGeneral::X_to_Xfloat( const size_t bucketsize ) {
    // kernel threads/blocks
    
    const size_t blocks = bucketsize / 32;
    const size_t threads = 32;
    kernel_X_to_Xfloat<<<blocks, threads, 0, stream>>>( dev_X, dev_X_float, VECDIM );
}

inline void GPUStreamGeneral::X_to_Xfloat_negate( const size_t bucketsize ) {
    // kernel threads/blocks
    
    const size_t blocks = bucketsize / 32;
    const size_t threads = 32;
    kernel_X_to_Xfloat_negate<<<blocks, threads, 0, stream>>>( dev_X, dev_ips, dev_X_float, VECDIM );
}

inline void GPUStreamGeneral::X_to_Xfloat_normalize( const size_t bucketsize ) {
    // kernel threads/blocks
    
    const size_t blocks = bucketsize / 32;
    const size_t threads = 32;
    kernel_X_to_Xfloat_normalize<<<blocks, threads, 0, stream>>>( dev_X, dev_len_in, dev_X_float, VECDIM );
}

inline void GPUStreamGeneral::float_to_half( const float* dev_in, half* dev_out, const size_t bucketsize ) {
    // kernel threads/blocks
    
    const size_t blocks = bucketsize / 32;
    const size_t threads = 32;
    kernel_float_to_half<<<blocks, threads, 0, stream>>>( dev_in, dev_out, VECDIM );
}

inline void GPUStreamGeneral::prepare_len_and_ips( float lenbound, float b_len, const size_t bucketsize ) {
    const size_t blocks = bucketsize / 128;
    const size_t threads = 32;
    
    //std::cerr << "Prepare using " << lenbound << " " << b_len << " " << bucketsize << std::endl;

    kernel_prepare_len_and_ips<<<blocks, threads, 0, stream>>>( dev_len_in, dev_ips, dev_len_half, lenbound, b_len );
}

inline void GPUStreamGeneral::reorder( half* dev, size_t nr_vecs ) {

    const size_t blocks = nr_vecs / 128;
    const size_t threads = 256;
    assert( nr_vecs % 128 == 0 );

    //std::cerr << "Reordering " << nr_vecs << " vecs using " << blocks << ", " << threads << std::endl;

    switch( VECDIM ) {
        case 32:
            kernel_reorder<32><<<blocks, threads, 0, stream>>>( dev, uint32_t(nr_vecs)); break;
        case 64:
            kernel_reorder<64><<<blocks, threads, 0, stream>>>( dev, uint32_t(nr_vecs)); break;
        case 96:
            kernel_reorder<96><<<blocks, threads, 0, stream>>>( dev, uint32_t(nr_vecs)); break;
        case 128:
            kernel_reorder<128><<<blocks, threads, 0, stream>>>( dev, uint32_t(nr_vecs)); break;
        case 160:
            kernel_reorder<160><<<blocks, threads, 0, stream>>>( dev, uint32_t(nr_vecs)); break;
        default:
            assert(false);
    }
    CUDA_CHECK( cudaPeekAtLastError() ); 
}


// ------------------------- BUCKETING ---------------------- //


void GPUStreamGeneral::B_init( const std::vector<size_t>& cdb_indices, const uint32_t _bucket_seed, const std::vector<double> &q, const int force_bdgl_blocks ) {

	curr_bucket = nullptr;
    prev_bucket = nullptr;
    cdb_range = {0,0};
    cdb_range_prev = {0,0}; 
    if( bucketer == "bgj1" ) {
        size_t real_nr_buckets = cdb_indices.size();
        b_cdb_indices = cdb_indices;
        assert( real_nr_buckets % 32 == 0 );

	nr_buckets = ((real_nr_buckets+255)/256) * 256;
	if (nr_buckets > max_nr_buckets)
		std::cerr << "B_init: Warning!: nr_buckets > max_nr_buckets" << std::endl;

        if( !global )
            return;

        send_X_range<true,true,true>( cdb_indices.begin(), cdb_indices.end() );
        X_to_Xfloat_normalize( real_nr_buckets );
        
        float alpha = 1;
        float beta = 0.;

        if (CUBLAS_STATUS_SUCCESS != cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, VECDIM, real_nr_buckets, VECDIM, &alpha, dev_mu_R, VECDIM, dev_X_float, VECDIM, &beta, dev_B_float,
VECDIM))
		std::cerr << "B_init(): cublasSgemm failed" << std::endl;
    
    	float_to_half( dev_B_float, dev_B, real_nr_buckets );

        // fill remaining buckets with 0.
    	if( nr_buckets > real_nr_buckets )
        	CUDA_CHECK( cudaMemsetAsync(dev_B + real_nr_buckets * VECDIM, 0, (nr_buckets-real_nr_buckets) * VECDIM * sizeof(half), stream) ); 

        reorder( dev_B, nr_buckets );

        // Wait for bucket to be ready
        CUDA_CHECK( cudaDeviceSynchronize() );


    } else if( bucketer == "bdgl" ) {
        nr_buckets = cdb_indices.size();
        bdgl_seed = _bucket_seed;
        
        // make sure this transitioning threshold is the same as in triple_sieve_gpu
        if( (force_bdgl_blocks==-1 and nr_buckets >= 8*1024) or force_bdgl_blocks==2 ) {
            bdgl_blocks = 2;
            bdgl_local_buckets = (size_t)(std::sqrt(nr_buckets/2)+0.5);
            assert( nr_buckets == 2 * bdgl_local_buckets * bdgl_local_buckets );
        } else {
            bdgl_blocks = 1;
            bdgl_local_buckets = nr_buckets;
        }
       
        if( !global )
            return;

        // OPTIONAL: RANDOM orthonormal transform
        const size_t q_size = VECDIM*VECDIM;
        std::memset( host_len_in, 0, (q_size)*sizeof(float));
    
        // q
        for( size_t i = 0; i < n; i++ ) {
            for( size_t j = 0; j < n; j++ ) {
                host_len_in[i * VECDIM+j] = (float)q[i*n+j];
            }
        }
        
        CUDA_CHECK( cudaMemcpyAsync(dev_q_transform, host_len_in, q_size * sizeof(float), cudaMemcpyHostToDevice, stream) ); 
    }

}

void GPUStreamGeneral::B_send_data( const size_t cdb_start, const size_t cdb_end ) {
	

    send_X_range<false>( cdb_start, cdb_end );
    X_to_Xfloat( cdb_end - cdb_start );

    CUDA_CHECK( cudaPeekAtLastError() ); 

    cdb_range_prev = cdb_range;
    cdb_range = {cdb_start, cdb_end};
}

void GPUStreamGeneral::B_launch_kernel() {
	
            // Compute blocks/threads
            const size_t nr_vecs = cdb_range.second - cdb_range.first;
            
            float alpha = 1.;
            float beta = 0.;
            
            if( bucketer == "bgj1" ) {
                cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, VECDIM, nr_vecs, VECDIM, &alpha, dev_mu_R, VECDIM, dev_X_float, VECDIM, &beta, dev_YR_float, VECDIM);
            }
            else if( bucketer == "bdgl" ) {
                cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, VECDIM, nr_vecs, VECDIM, &alpha, dev_q_transform, VECDIM, dev_X_float, VECDIM, &beta, dev_YR_float, VECDIM);
            }


            float_to_half( dev_YR_float, dev_YR_half, nr_vecs );
            
            if( bucketer == "bgj1" ) {
                reorder( dev_YR_half, nr_vecs );

                const size_t blocks = nr_vecs / 128;
                const size_t threads = 256;

                switch( multi_bucket ) {
                    case 1:
                        switch( VECDIM ) {
                            case 32:
                                kernel_bucketing<32, 1><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break; 
                            case 64:
                                kernel_bucketing<64, 1><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 96:
                                kernel_bucketing<96, 1><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 128:
                                kernel_bucketing<128, 1><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 160:
                                kernel_bucketing<160, 1><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            default:
                                assert(false);
                        }
                        break;
                    case 2:
                        switch( VECDIM ) {
                            case 32:
                                kernel_bucketing<32, 2><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break; 
                            case 64:
                                kernel_bucketing<64, 2><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 96:
                                kernel_bucketing<96, 2><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 128:
                                kernel_bucketing<128, 2><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 160:
                                kernel_bucketing<160, 2><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            default:
                                assert(false);
                        }
                        break;
                    case 4:
                        switch( VECDIM ) {
                            case 32:
                                kernel_bucketing<32, 4><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break; 
                            case 64:
                                kernel_bucketing<64, 4><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 96:
                                kernel_bucketing<96, 4><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 128:
                                kernel_bucketing<128, 4><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 160:
                                kernel_bucketing<160, 4><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            default:
                                assert(false);
                        }
                        break;
                    case 8:
                        switch( VECDIM ) {
                            case 32:
                                kernel_bucketing<32, 8><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break; 
                            case 64:
                                kernel_bucketing<64, 8><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 96:
                                kernel_bucketing<96, 8><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 128:
                                kernel_bucketing<128, 8><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 160:
                                kernel_bucketing<160, 8><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            default:
                                assert(false);
                        }
                        break;
                    case 16:
                        switch( VECDIM ) {
                            case 32:
                                kernel_bucketing<32, 16><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break; 
                            case 64:
                                kernel_bucketing<64, 16><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 96:
                                kernel_bucketing<96, 16><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 128:
                                kernel_bucketing<128, 16><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            case 160:
                                kernel_bucketing<160, 16><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_B, uint32_t(nr_buckets), dev_indices, dev_ips ); break;
                            default:
                                assert(false);
                        }
                        break;
                    default:
                        assert(false);
                }
            }
            else if( bucketer == "bdgl" ) {
                const size_t blocks = 512;
                const size_t threads = 32;
                
				assert( bdgl_blocks == 1 or bdgl_blocks == 2);
				assert( multi_bucket == 1 or multi_bucket == 2 or multi_bucket == 4 or multi_bucket == 8 or multi_bucket == 16);

				switch( bdgl_blocks ) {
					case 1:
					switch( multi_bucket ) {
						case 1: switch( VECDIM ) {
								case 64: kernel_bdgl_bucketing_1block<64,1><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 96: kernel_bdgl_bucketing_1block<96,1><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_1block<128,1><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: assert(false); break;
							} break;
						case 2: switch( VECDIM ) {
								case 64: kernel_bdgl_bucketing_1block<64,2><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 96: kernel_bdgl_bucketing_1block<96,2><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_1block<128,2><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: assert(false); break;
							} break;
                        case 4: switch( VECDIM ) {
								case 64: kernel_bdgl_bucketing_1block<64,4><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 96: kernel_bdgl_bucketing_1block<96,4><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_1block<128,4><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: assert(false); break;
							} break;
                        case 8: switch( VECDIM ) {
								case 64: kernel_bdgl_bucketing_1block<64,8><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 96: kernel_bdgl_bucketing_1block<96,8><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_1block<128,8><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: assert(false); break;
							} break;
                        case 16: switch( VECDIM ) {
								case 64: kernel_bdgl_bucketing_1block<64,16><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 96: kernel_bdgl_bucketing_1block<96,16><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_1block<128,16><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: assert(false); break;
							} break;

					} break;
					case 2:
					switch( multi_bucket ) {
						case 1: switch( VECDIM ) {
								case 64: assert(false); break;
								case 96: kernel_bdgl_bucketing_2block<96,1><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_2block<128,1><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: kernel_bdgl_bucketing_2block<160,1><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
							} break;
						case 2: switch( VECDIM ) {
								case 64: assert(false); break;
								case 96: kernel_bdgl_bucketing_2block<96,2><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_2block<128,2><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: kernel_bdgl_bucketing_2block<160,2><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
							} break;
                        case 4: switch( VECDIM ) {
								case 64: assert(false); break;
								case 96: kernel_bdgl_bucketing_2block<96,4><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_2block<128,4><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: kernel_bdgl_bucketing_2block<160,4><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
							} break;
                        case 8: switch( VECDIM ) {
								case 64: assert(false); break;
								case 96: kernel_bdgl_bucketing_2block<96,8><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_2block<128,8><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: kernel_bdgl_bucketing_2block<160,8><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
							} break;
                        case 16: switch( VECDIM ) {
								case 64: assert(false); break;
								case 96: kernel_bdgl_bucketing_2block<96,16><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 128: kernel_bdgl_bucketing_2block<128,16><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
								case 160: kernel_bdgl_bucketing_2block<160,16><<<blocks, threads>>>(dev_YR_half, (int32_t)n, (uint32_t)nr_vecs, (int)bdgl_local_buckets, bdgl_seed, dev_indices, dev_ips); break;
							} break;
					} break;
				}                
           }

            CUDA_CHECK( cudaPeekAtLastError() );    
    }

void GPUStreamGeneral::B_receive_data(std::vector<triple_bucket> &bucket, const size_t max_vecs_per_bucket, bool onlyprocess) {

            // wait for data to arrive
            CUDA_CHECK( cudaEventSynchronize( D2H ) );
            
            const size_t cdb_start = onlyprocess ? cdb_range.first : cdb_range_prev.first;
            const size_t cdb_end = onlyprocess ? cdb_range.second : cdb_range_prev.second;
            if( cdb_end - cdb_start > 0 ) { 
                for( indextype i = 0; i < cdb_end-cdb_start; i++ ) {
                    indextype db_index = cdb[cdb_start+i].i;
                    for( indextype j = 0; j < multi_bucket; j++ ) {
                        indextype b_index = host_indices[multi_bucket*i+j];
                        auto offset = bucket[b_index].size++;
                        if( offset < max_vecs_per_bucket ) {
                            bucket[b_index].indices[offset] = db_index;
                            bucket[b_index].ips[offset] = host_ips[multi_bucket*i+j];
                        }
                        if( bucketer == "bgj1" )
							bucket[b_index].size -= ( b_cdb_indices[b_index] == cdb_start+i ) ? 1 : 0;
					}
                }
            }
            if( !onlyprocess ) {
                const size_t results = (cdb_range.second - cdb_range.first) * multi_bucket;

                // retrieve new results
                CUDA_CHECK( cudaMemcpyAsync(host_indices, dev_indices, results * sizeof(indextype) , cudaMemcpyDeviceToHost, stream) );
                CUDA_CHECK( cudaMemcpyAsync(host_ips, dev_ips, results * sizeof(iptype), cudaMemcpyDeviceToHost, stream) );

                // triggers when results retrieved
                CUDA_CHECK( cudaEventRecord(D2H, stream));
            }
    }

// --------------------------- PROCESSING ------------------------ //

void GPUStreamGeneral::P_send_data( const triple_bucket &bucket, const float lenbound, const float dh_bucket_ratio ) {
    //std::cerr << "Start send data with bucketing of size " << bucket.indices.size() << std::endl;

    prev_bucket = curr_bucket;
    curr_bucket = &bucket;

    size_t bucketsize = bucket.size;
    bucketsize = std::min( bucketsize, size_t(VECNUM));
    bucketsize -= bucketsize % 128;
    assert( bucketsize % 128 == 0 );
        
    send_X_range<true, false, false>( bucket.indices, bucket.indices + bucketsize ); 

    memcpy( host_ips, bucket.ips, bucketsize * sizeof(iptype) );
    CUDA_CHECK( cudaMemcpyAsync(dev_ips, host_ips, bucketsize * sizeof(iptype), cudaMemcpyHostToDevice, stream) ); 

    CUDA_CHECK( cudaEventRecord( H2D, stream ) );

    X_to_Xfloat_negate( bucketsize );
    float alpha = 1;
    float beta = 0.;

    // compute dual hash, use YR_half and YR_float for intermediate data storage
    if( lift ) {

        size_t lift_bucketsize = dh_bucket_ratio * bucket.size;
        lift_bucketsize = std::min( lift_bucketsize, size_t(VECNUM/2));
        lift_bucketsize -= lift_bucketsize % 128;

        // compute YR on lift context
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, MAX_LIFT, lift_bucketsize, VECDIM, &alpha, dev_mu_L, VECDIM, dev_X_float, VECDIM, &beta, (float*)dev_YR_half, MAX_LIFT);

        // compute DH
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dual_hash_vecs, lift_bucketsize, MAX_LIFT, &alpha, dev_dual_hash, MAX_LIFT, (float*)dev_YR_half, MAX_LIFT, &beta, dev_YR_float, dual_hash_vecs);
        // Convert to 8bit
        const size_t blocks = lift_bucketsize / 32;
        const size_t threads = 32;

        switch( dual_hash_vecs ) {
            case 16:
                kernel_float_to_8bit_transpose<16><<<blocks, threads, 0, stream>>>( dev_YR_float, dev_DH );
                break;
            case 32:
                kernel_float_to_8bit_transpose<32><<<blocks, threads, 0, stream>>>( dev_YR_float, dev_DH );
                break;
            case 48:
                kernel_float_to_8bit_transpose<48><<<blocks, threads, 0, stream>>>( dev_YR_float, dev_DH );
                break;
            case 64:
                kernel_float_to_8bit_transpose<64><<<blocks, threads, 0, stream>>>( dev_YR_float, dev_DH );
                break;
            default:
                assert(false);
        }
        last_lift_bucketsize = lift_bucketsize;
    }


    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, VECDIM, bucketsize, VECDIM, &alpha, dev_mu_R, VECDIM, dev_X_float, VECDIM, &beta, dev_YR_float, VECDIM);
    float_to_half( dev_YR_float, dev_YR_half, bucketsize );
    
    // Prepare lengths and ips
    prepare_len_and_ips( lenbound, bucket.b_len, bucketsize );

    //kernel_print<<<1, 32, 0, stream>>>( dev_len_half, 64 );
        
    // Reorder data
    reorder( dev_YR_half, bucketsize ); 

    // number of results with cpu
    /*int res = 0;
    for( int i = 0; i < bucketsize; i++ ) {
        int startj = ((i+128)/128)*128;
        int endj = bucketsize - ((int(bucketsize)-startj)%256);
        for( int j = startj; j < endj; j++ ) {
            float sm = 0.;
            for( int k = 0; k < VECDIM; k++ ) {
                sm += db[bucket.indices[i]].yr[k] * db[bucket.indices[j]].yr[k];
            }
            if( (__half2float(bucket.ips[i])>0.)^(__half2float(bucket.ips[j])>0.) )  
                res += (db[bucket.indices[i]].len + db[bucket.indices[j]].len + 2 *  sm ) < lenbound;
            else 
                res += (db[bucket.indices[i]].len + db[bucket.indices[j]].len - 2 *  sm ) < lenbound;
        }
    
    }
    std::cerr << "NR results cpu: " << res << std::endl;*/

    last_bucketsize = bucketsize;
}

void GPUStreamGeneral::P_launch_kernel( uint32_t dh_bound ) {
            
        //std::cerr << "Launch Kernel" << std::endl;

        auto B_id = curr_bucket->b_local_index;
        lentype B_len = curr_bucket->b_len;

        CUDA_CHECK( cudaMemsetAsync(dev_nr_results, 0, 2*sizeof(indextype), stream) );
        // Compute blocks/threads
        const size_t blocks = last_bucketsize / 128;
        const size_t threads = 256;


        if( benchmark ) {
            bench_sieve_flop += float(last_bucketsize) * float(last_bucketsize) * VECDIM;
            CUDA_CHECK( cudaEventRecord(start[bench_alternation], stream));
        }

		switch( VECDIM ) {
            case 32:
                if( triple )
                    kernel_triple_sieve<32,true><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                else
                    kernel_triple_sieve<32,false><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                if( benchmark ) CUDA_CHECK( cudaEventRecord(stop[bench_alternation], stream));
                if( triple )
                    kernel_postprocess<32,true><<<128, 32, 0, stream>>>( dev_YR_float, dev_B_float + VECDIM * B_id, B_len, dev_indices, dev_nr_results, dev_len_out);
                else
                    kernel_postprocess<32,false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, B_len, dev_indices, dev_nr_results, dev_len_out);
                break;
            case 64:  
                if( triple )
                    kernel_triple_sieve<64,true><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                else
                    kernel_triple_sieve<64,false><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                if( benchmark ) CUDA_CHECK( cudaEventRecord(stop[bench_alternation], stream));
                if( triple )
                    kernel_postprocess<64,true><<<128, 32, 0, stream>>>( dev_YR_float, dev_B_float + VECDIM * B_id, B_len, dev_indices, dev_nr_results, dev_len_out);
                else
                    kernel_postprocess<64,false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, B_len, dev_indices, dev_nr_results, dev_len_out);
                break;
            case 96:  
                if( triple )
                    kernel_triple_sieve<96,true><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                else
                    kernel_triple_sieve<96,false><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                if( benchmark ) CUDA_CHECK( cudaEventRecord(stop[bench_alternation], stream));
                if( triple )
                    kernel_postprocess<96,true><<<128, 32, 0, stream>>>( dev_YR_float, dev_B_float + VECDIM * B_id, B_len, dev_indices, dev_nr_results, dev_len_out);
                else
                    kernel_postprocess<96,false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, B_len, dev_indices, dev_nr_results, dev_len_out);
                break;
             case 128:  
                if( triple )
                    kernel_triple_sieve<128,true><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                else
                    kernel_triple_sieve<128,false><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                if( benchmark ) CUDA_CHECK( cudaEventRecord(stop[bench_alternation], stream));
                if( triple )
                    kernel_postprocess<128,true><<<128, 32, 0, stream>>>( dev_YR_float, dev_B_float + VECDIM * B_id, B_len, dev_indices, dev_nr_results, dev_len_out);
                else
                    kernel_postprocess<128,false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, B_len, dev_indices, dev_nr_results, dev_len_out);
                break;
            case 160:  
                if( triple )
                    kernel_triple_sieve<160,true><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                else
                    kernel_triple_sieve<160,false><<<blocks, threads, 0, stream>>>( dev_YR_half, dev_len_half, dev_ips, last_bucketsize, dev_nr_results, (int*)dev_indices );
                if( benchmark ) CUDA_CHECK( cudaEventRecord(stop[bench_alternation], stream));
                if( triple )
                    kernel_postprocess<160,true><<<128, 32, 0, stream>>>( dev_YR_float, dev_B_float + VECDIM * B_id, B_len, dev_indices, dev_nr_results, dev_len_out);
                else
                    kernel_postprocess<160,false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, B_len, dev_indices, dev_nr_results, dev_len_out);
                break;
            default:
                assert(false);
        } 

        if( lift and dh_bound > 0 ) {
            dim3 gridDim, blockDim;

            gridDim.x = last_lift_bucketsize/128; // assume integer
            gridDim.y = 1; // assume integer

            blockDim.x = 8;
            blockDim.y = 16; 
            
            switch( dual_hash_vecs ) {
                case 16:
                    kernel_dualhash<16><<<gridDim, blockDim, 0, stream>>>( (uint*)dev_DH, dh_bound, dev_lift_indices, &(dev_nr_results[1]) ); break;
                case 32:
                    kernel_dualhash<32><<<gridDim, blockDim, 0, stream>>>( (uint*)dev_DH, dh_bound, dev_lift_indices, &(dev_nr_results[1]) ); break;
                case 48:
                    kernel_dualhash<48><<<gridDim, blockDim, 0, stream>>>( (uint*)dev_DH, dh_bound, dev_lift_indices, &(dev_nr_results[1]) ); break;
                case 64:
                    kernel_dualhash<64><<<gridDim, blockDim, 0, stream>>>( (uint*)dev_DH, dh_bound, dev_lift_indices, &(dev_nr_results[1]) ); break;
                default:
                    assert(false);
            }

            switch( VECDIM ) {
                case 32:
                    kernel_postprocess<32, false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, 0., dev_lift_indices, &(dev_nr_results[1]), dev_lift_len_out);
                    break;
                case 64:
                    kernel_postprocess<64, false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, 0., dev_lift_indices, &(dev_nr_results[1]), dev_lift_len_out);
                    break;
                case 96:
                    kernel_postprocess<96, false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, 0., dev_lift_indices, &(dev_nr_results[1]), dev_lift_len_out);
                    break;
                case 128:
                    kernel_postprocess<128, false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, 0., dev_lift_indices, &(dev_nr_results[1]), dev_lift_len_out);
                    break;
                case 160:
                    kernel_postprocess<160, false><<<128, 32, 0, stream>>>( dev_YR_float, nullptr, 0., dev_lift_indices, &(dev_nr_results[1]), dev_lift_len_out);
                    break;
                default:
                    assert(false);
            }

        } else if( lift and dh_bound == 0 ) {
            CUDA_CHECK( cudaMemsetAsync( &(dev_nr_results[1]), 0, sizeof(indextype), stream));
        }


        //CUDA_CHECK( cudaPeekAtLastError() );    
    
}

void GPUStreamGeneral::P_receive_data( queues &queue, bool onlyprocess) {
            // wait for data to arrive
            CUDA_CHECK( cudaEventSynchronize( D2H ) );

            if( benchmark )
                bench_alternation = !bench_alternation;

            const uint32_t max_results = VECNUM;
            if( prev_bucket != nullptr or (onlyprocess and curr_bucket != nullptr)  ) {
                
                if( benchmark ) {
                    float milliseconds = 0;
                    CUDA_CHECK( cudaEventElapsedTime(&milliseconds, start[bench_alternation], stop[bench_alternation]));
                    bench_sieve_kernel += milliseconds;
                }

                const indextype* bucket_indices = onlyprocess ? curr_bucket->indices : prev_bucket->indices;
                const iptype* bucket_ips = onlyprocess ? curr_bucket->ips : prev_bucket->ips;
                const indextype b_index = onlyprocess ? curr_bucket->b_db_index : prev_bucket->b_db_index;
                const auto b_size = onlyprocess ? curr_bucket->size : prev_bucket->size;

                if( host_nr_results[0] > max_results ) {
                    std::cerr << "Result overflow " << host_nr_results[0] << std::endl;
                    host_nr_results[0] = max_results;
                }

                for( size_t i = 0; i < host_nr_results[0]; i++ ) {
                    int i1 = ((int*)host_indices)[2*i];
                    int i2 = ((int*)host_indices)[2*i+1];

                    uint8_t sign1 = __half2float(bucket_ips[std::abs(i1)]) > 0.f;
                    uint8_t sign2 = __half2float(bucket_ips[std::abs(i2)]) > 0.f;
                    
                    if( i1 >= 0 and i2 >= 0 ) {
                        uint8_t sign = uint8_t(1)^sign1^sign2;
                        queue.sieve_pairs.push_back( { { bucket_indices[i1], bucket_indices[i2] }, 
                                                    host_len_out[i],
                                                    sign
                                                    } );
                    } else {
                        assert( i1 <= 0 and i2 <= 0 );
                        uint8_t sign = sign1|(sign2<<1);
                        queue.sieve_triples.push_back( { { b_index, bucket_indices[-i1], bucket_indices[-i2] },
                                                        host_len_out[i],
                                                        sign
                        });
                    }
                }
                
                if( lift ) {
                    if( host_nr_results[1] > 0.9 * max_results ) 
                        std::cerr << "Close to overflow " << host_nr_results[1] << std::endl;

                    if( host_nr_results[1] > max_results ) {
                        host_nr_results[1] = max_results;
                    }


                    for( size_t i = 0; i < host_nr_results[1]; i++ ) {
                        int i1 = ((int*)host_lift_indices)[2*i];
                        int i2 = ((int*)host_lift_indices)[2*i+1];
                        
                        //std::cerr << i1 << " " << i2 << std::endl;

                        uint8_t sign1 = __half2float(bucket_ips[std::abs(i1)]) > 0.f;
                        uint8_t sign2 = __half2float(bucket_ips[std::abs(i2)]) > 0.f;              

                        uint8_t sign = uint8_t(1)^sign1^sign2;
                        queue.lift_pairs.push_back( { { bucket_indices[i1], bucket_indices[i2] },
                                                        host_lift_len_out[i],
                                                        sign                    
                                                    } );        
                    }

                }

                (onlyprocess ? curr_bucket : prev_bucket) = nullptr;
            }

            if( !onlyprocess ) {
                const size_t results = max_results;
                // retrieve new results
                CUDA_CHECK( cudaMemcpyAsync(host_indices, dev_indices, results * 2 * sizeof(indextype) , cudaMemcpyDeviceToHost, stream) );
                CUDA_CHECK( cudaMemcpyAsync(host_len_out, dev_len_out, results * sizeof(lentype), cudaMemcpyDeviceToHost, stream) );
                
                if( lift ) {
                    CUDA_CHECK( cudaMemcpyAsync(host_lift_indices, dev_lift_indices, results * 2 * sizeof(indextype), cudaMemcpyDeviceToHost, stream) );
                    CUDA_CHECK( cudaMemcpyAsync(host_lift_len_out, dev_lift_len_out, results * sizeof(lentype), cudaMemcpyDeviceToHost, stream) );
                }


                CUDA_CHECK( cudaMemcpyAsync(host_nr_results, dev_nr_results, 2*sizeof(indextype), cudaMemcpyDeviceToHost, stream) );
                
                // triggers when results retrieved
                CUDA_CHECK( cudaEventRecord(D2H, stream));
            }
    }

// ------------------------- EXTEND LEFT BABAI --------------------- //

void GPUStreamGeneral::E_send_data( const size_t db_start, const size_t db_end ) {
    size_t bucketsize = db_end - db_start;
    assert( bucketsize%32==0);
    
    send_X_range<false, false>( db_start, db_end );

    cdb_range_prev = cdb_range;
    cdb_range = {db_start, db_end};
    
    X_to_Xfloat( bucketsize );
    float alpha = 1;
    float beta = 0.;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, VECDIM, bucketsize, VECDIM, &alpha, dev_mu_R, VECDIM, dev_X_float, VECDIM, &beta, dev_YR_float, VECDIM);
}

void GPUStreamGeneral::E_launch_kernel( const size_t extend_left ) {
    // Compute blocks/threads
    const size_t blocks = 128;
    const size_t threads = 32;
    
    size_t bucketsize = cdb_range.second - cdb_range.first;
    
    assert( extend_left < MAX_EXTEND );

    // babai rounding on extend_left first coordinates
    switch( extend_left ) {
        case 1:
            kernel_babai<1><<<blocks, threads, 0, stream>>>( dev_X, dev_YR_float, dev_X_extend, dev_mu_R, VECDIM, bucketsize); break;
        case 2:  
            kernel_babai<2><<<blocks, threads, 0, stream>>>( dev_X, dev_YR_float, dev_X_extend, dev_mu_R, VECDIM, bucketsize); break;
        case 3:  
            kernel_babai<3><<<blocks, threads, 0, stream>>>( dev_X, dev_YR_float, dev_X_extend, dev_mu_R, VECDIM, bucketsize); break;
         case 4:  
            kernel_babai<4><<<blocks, threads, 0, stream>>>( dev_X, dev_YR_float, dev_X_extend, dev_mu_R, VECDIM, bucketsize); break;
        default:
            assert(false);
    }
    CUDA_CHECK( cudaPeekAtLastError() );    
   
    // recompute lengths
    switch( VECDIM ) {
        case 32:
            kernel_recompute_len<32><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize ); 
            kernel_recompute_uid<32><<<128, 32, 0, stream>>>( dev_X, (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
        case 64:  
            kernel_recompute_len<64><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<64><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
        case 96: 
            kernel_recompute_len<96><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<96><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
         case 128: 
            kernel_recompute_len<128><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<128><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
        case 160: 
            kernel_recompute_len<160><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<160><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break; 
        default:
            assert(false);
    }
    CUDA_CHECK( cudaPeekAtLastError() );

}

void GPUStreamGeneral::E_receive_data( const size_t extend_left, bool onlyprocess) {
            
    // wait for data to arrive
    CUDA_CHECK( cudaEventSynchronize( D2H ) );
    
    const size_t db_start = onlyprocess ? cdb_range.first : cdb_range_prev.first;
    const size_t db_end = onlyprocess ? cdb_range.second : cdb_range_prev.second;

    UidType* host_uid = reinterpret_cast<UidType*>(host_indices); 
    
    for( size_t i = 0; i < db_end-db_start; i++ ) {
        Entry* e = &db[db_start+i];
        for( size_t j = 0; j < extend_left; j++ )
            e->x[j] += host_X_extend[extend_left*i+j];
        e->len = host_len_out[i];
        e->uid = host_uid[i];
    }

    if( !onlyprocess ) {
        const size_t results = (cdb_range.second - cdb_range.first);

        // retrieve new results 
        CUDA_CHECK( cudaMemcpyAsync(host_X_extend, dev_X_extend, results * extend_left * sizeof(Xtype) , cudaMemcpyDeviceToHost, stream) );
        CUDA_CHECK( cudaMemcpyAsync(host_len_out, dev_len_out, results * sizeof(lentype), cudaMemcpyDeviceToHost, stream) );
        // uid reusing _indices
        CUDA_CHECK( cudaMemcpyAsync(host_indices, dev_indices, results * sizeof(UidType), cudaMemcpyDeviceToHost, stream) );
        // triggers when results retrieved
        CUDA_CHECK( cudaEventRecord(D2H, stream));
    } else {
        cdb_range_prev = {0,0};
        cdb_range = {0,0};
    }
}

// ------------------------- RECOMPUTE LEN --------------------- //


void GPUStreamGeneral::R_send_data( const size_t db_start, const size_t db_end ) {
    size_t bucketsize = db_end - db_start;
    assert( bucketsize%32==0);
    
    send_X_range<false, false>( db_start, db_end );

    cdb_range_prev = cdb_range;
    cdb_range = {db_start, db_end};
    
    X_to_Xfloat( bucketsize );
    float alpha = 1;
    float beta = 0.;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, VECDIM, bucketsize, VECDIM, &alpha, dev_mu_R, VECDIM, dev_X_float, VECDIM, &beta, dev_YR_float, VECDIM);

}

void GPUStreamGeneral::R_launch_kernel( ) {
            
    size_t bucketsize = cdb_range.second - cdb_range.first;
    switch( VECDIM ) {
        case 32:
            kernel_recompute_len<32><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize ); 
            kernel_recompute_uid<32><<<128, 32, 0, stream>>>( dev_X, (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
        case 64:  
            kernel_recompute_len<64><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<64><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
        case 96: 
            kernel_recompute_len<96><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<96><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
         case 128: 
            kernel_recompute_len<128><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<128><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break;
        case 160: 
            kernel_recompute_len<160><<<128, 32, 0, stream>>>( dev_YR_float, dev_len_out, bucketsize );  
            kernel_recompute_uid<160><<<128, 32, 0, stream>>>( dev_X,  (UidType*)dev_indices, dev_uid_coeffs, bucketsize ); break; 
        default:
            assert(false);
    }
    CUDA_CHECK( cudaPeekAtLastError() );    
}

void GPUStreamGeneral::R_receive_data(bool onlyprocess) {

            // wait for data to arrive
            CUDA_CHECK( cudaEventSynchronize( D2H ) );
            
            const size_t db_start = onlyprocess ? cdb_range.first : cdb_range_prev.first;
            const size_t db_end = onlyprocess ? cdb_range.second : cdb_range_prev.second;
            
            UidType* host_uid = reinterpret_cast<UidType*>(host_indices); 

            for( size_t i = 0; i < db_end-db_start; i++ ) {
                Entry* e = &db[db_start+i];
                e->len = host_len_out[i];
                e->uid = host_uid[i];
            }

            if( !onlyprocess ) {
                const size_t results = (cdb_range.second - cdb_range.first);

                // retrieve new results
                CUDA_CHECK( cudaMemcpyAsync(host_len_out, dev_len_out, results * sizeof(lentype), cudaMemcpyDeviceToHost, stream) );
                // uid reusing _indices
                CUDA_CHECK( cudaMemcpyAsync(host_indices, dev_indices, results * sizeof(UidType), cudaMemcpyDeviceToHost, stream) );

                // triggers when results retrieved
                CUDA_CHECK( cudaEventRecord(D2H, stream));
            } else {
                cdb_range_prev = {0,0};
                cdb_range = {0,0};
            }
}

void GPUStreamGeneral::sync() {
    CUDA_CHECK( cudaStreamSynchronize(stream) );
}
