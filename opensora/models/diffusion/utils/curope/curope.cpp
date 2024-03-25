/* 
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <torch/extension.h>

// forward declaration
void rope_2d_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd );

void rope_2d_cpu( torch::Tensor tokens, const torch::Tensor positions, const float base, const float fwd )
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3) / 4;

    auto tok = tokens.accessor<float, 4>();
    auto pos = positions.accessor<int64_t, 3>();

    for (int b = 0; b < B; b++) {
      for (int x = 0; x < 2; x++) { // y and then x (2d)
        for (int n = 0; n < N; n++) {
        
            // grab the token position
            const int p = pos[b][n][x];

            for (int h = 0; h < H; h++) {
                for (int d = 0; d < D; d++) {
                    // grab the two values
                    float u = tok[b][n][h][d+0+x*2*D];
                    float v = tok[b][n][h][d+D+x*2*D];

                    // grab the cos,sin
                    const float inv_freq = fwd * p / powf(base, d/float(D));
                    float c = cosf(inv_freq);
                    float s = sinf(inv_freq);

                    // write the result
                    tok[b][n][h][d+0+x*2*D] = u*c - v*s;
                    tok[b][n][h][d+D+x*2*D] = v*c + u*s;
                }
            }
        }
      }
    }
}

void rope_2d( torch::Tensor tokens,     // B,N,H,D
        const torch::Tensor positions,  // B,N,2
        const float base, 
        const float fwd )
{
    TORCH_CHECK(tokens.dim() == 4, "tokens must have 4 dimensions");
    TORCH_CHECK(positions.dim() == 3, "positions must have 3 dimensions");
    TORCH_CHECK(tokens.size(0) == positions.size(0), "batch size differs between tokens & positions");
    TORCH_CHECK(tokens.size(1) == positions.size(1), "seq_length differs between tokens & positions");
    TORCH_CHECK(positions.size(2) == 2, "positions.shape[2] must be equal to 2");
    TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(), "tokens and positions are not on the same device" );

    if (tokens.is_cuda())
        rope_2d_cuda( tokens, positions, base, fwd );
    else
        rope_2d_cpu( tokens, positions, base, fwd );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_2d", &rope_2d, "RoPE 2d forward/backward");
}
