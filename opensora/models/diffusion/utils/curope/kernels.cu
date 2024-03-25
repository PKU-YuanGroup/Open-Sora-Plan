/* 
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(tensor) {\
    TORCH_CHECK((tensor).is_cuda(), #tensor " is not in cuda memory"); \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " is not contiguous"); }
void CHECK_KERNEL() {auto error = cudaGetLastError(); TORCH_CHECK( error == cudaSuccess, cudaGetErrorString(error));}


template < typename scalar_t  >
__global__ void rope_2d_cuda_kernel( 
        //scalar_t* __restrict__ tokens, 
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> tokens,
        const int64_t* __restrict__ pos, 
        const float base, 
        const float fwd )
        // const int N, const int H, const int D )
{
    // tokens shape = (B, N, H, D)
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3);
    
    // each block update a single token, for all heads
    // each thread takes care of a single output
    extern __shared__ float shared[];
    float* shared_inv_freq = shared + D;

    const int b = blockIdx.x / N;
    const int n = blockIdx.x % N;

    const int Q = D / 4; 
    // one token = [0..Q : Q..2Q : 2Q..3Q : 3Q..D]
    //              u_Y     v_Y     u_X      v_X

    // shared memory: first, compute inv_freq
    if (threadIdx.x < Q)
        shared_inv_freq[threadIdx.x] = fwd / powf(base, threadIdx.x/float(Q));
    __syncthreads();

    // start of X or Y part
    const int X = threadIdx.x < D/2 ? 0 : 1; 
    const int m = (X*D/2) + (threadIdx.x % Q);   // index of u_Y or u_X

    // grab the cos,sin appropriate for me
    const float freq = pos[blockIdx.x*2+X] * shared_inv_freq[threadIdx.x % Q];
    const float cos = cosf(freq);
    const float sin = sinf(freq);
    /*
    float* shared_cos_sin = shared + D + D/4;
    if ((threadIdx.x % (D/2)) < Q)
        shared_cos_sin[m+0] = cosf(freq);
    else
        shared_cos_sin[m+Q] = sinf(freq);
    __syncthreads();
    const float cos = shared_cos_sin[m+0];
    const float sin = shared_cos_sin[m+Q];
    */

    for (int h = 0; h < H; h++)
    {
        // then, load all the token for this head in shared memory
        shared[threadIdx.x] = tokens[b][n][h][threadIdx.x];
        __syncthreads();

        const float u = shared[m];
        const float v = shared[m+Q];
        
        // write output
        if ((threadIdx.x % (D/2)) < Q)
            tokens[b][n][h][threadIdx.x] = u*cos - v*sin;
        else
            tokens[b][n][h][threadIdx.x] = v*cos + u*sin;
    }
}

void rope_2d_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd ) 
{
    const int B = tokens.size(0); // batch size
    const int N = tokens.size(1); // sequence length
    const int H = tokens.size(2); // number of heads
    const int D = tokens.size(3); // dimension per head

    TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D, "tokens are not contiguous");
    TORCH_CHECK(pos.is_contiguous(), "positions are not contiguous");
    TORCH_CHECK(pos.size(0) == B && pos.size(1) == N && pos.size(2) == 2, "bad pos.shape");
    TORCH_CHECK(D % 4 == 0, "token dim must be multiple of 4");

    // one block for each layer, one thread per local-max
    const int THREADS_PER_BLOCK = D;
    const int N_BLOCKS = B * N; // each block takes care of H*D values
    const int SHARED_MEM = sizeof(float) * (D + D/4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
        rope_2d_cuda_kernel<scalar_t> <<<N_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM>>> (
            //tokens.data_ptr<scalar_t>(), 
            tokens.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            pos.data_ptr<int64_t>(), 
            base, fwd); //, N, H, D );
    }));
}
