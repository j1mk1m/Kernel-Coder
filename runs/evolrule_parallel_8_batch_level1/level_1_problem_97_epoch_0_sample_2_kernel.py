import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for computing attention scores (Q*K^T / sqrt(D))
attention_scores_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void compute_attention_scores(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    scalar_t* __restrict__ scores,
    int B, int H, int S, int D) {
    const int tile_size = 32;
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int tile_idx = blockIdx.y;
    int num_tiles = (S + tile_size - 1) / tile_size;
    int tile_row = tile_idx / num_tiles;
    int tile_col = tile_idx % num_tiles;
    int row_start = tile_row * tile_size;
    int col_start = tile_col * tile_size;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_in_tile = ty;
    int col_in_tile = tx;
    int global_row = row_start + row_in_tile;
    int global_col = col_start + col_in_tile;

    __shared__ scalar_t shared_Q[tile_size][tile_size];
    __shared__ scalar_t shared_K[tile_size][tile_size];

    scalar_t sum = 0.0;

    for (int k = 0; k < D; k += tile_size) {
        // Load Q tile
        int Q_offset = b * H * S * D + h * S * D;
        int Q_row = global_row;
        int Q_col = k + col_in_tile;
        if (Q_row < S && Q_col < D) {
            shared_Q[row_in_tile][col_in_tile] = Q[Q_offset + Q_row * D + Q_col];
        } else {
            shared_Q[row_in_tile][col_in_tile] = 0.0;
        }

        // Load K tile (transposed)
        int K_offset = b * H * S * D + h * S * D;
        int K_row = k + row_in_tile;
        int K_col = global_col;
        if (K_row < D && K_col < S) {
            shared_K[row_in_tile][col_in_tile] = K[K_offset + K_col * D + K_row];
        } else {
            shared_K[row_in_tile][col_in_tile] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < tile_size; ++i) {
            sum += shared_Q[row_in_tile][i] * shared_K[i][col_in_tile];
        }

        __syncthreads();
    }

    if (global_row < S && global_col < S) {
        int score_offset = b * H * S * S + h * S * S + global_row * S + global_col;
        scores[score_offset] = sum / sqrt(D);
    }
}

std::tuple<torch::Tensor> attention_scores(
    torch::Tensor Q,
    torch::Tensor K) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto S = Q.size(2);
    const auto D = Q.size(3);

    auto scores = torch::zeros({B, H, S, S}, Q.options());

    const int tile_size = 32;
    const int num_tiles = (S + tile_size - 1) / tile_size;
    const dim3 threads(tile_size, tile_size);
    const dim3 blocks(B * H, num_tiles * num_tiles);

    AT_DISPATCH_FLOATING_TYPES(Q.scalar_type(), "compute_attention_scores", ([&] {
        compute_attention_scores<scalar_t><<<blocks, threads>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            scores.data_ptr<scalar_t>(),
            B, H, S, D);
    }));

    return std::make_tuple(scores);
}
"""

# Custom CUDA kernel for applying softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void softmax_kernel(
    scalar_t* scores,
    int B, int H, int S) {
    int b = blockIdx.x / (H * S);
    int h = (blockIdx.x / S) % H;
    int s = blockIdx.x % S;
    int tid = threadIdx.x;

    extern __shared__ scalar_t shared_data[];
    if (tid < S) {
        int offset = b * H * S * S + h * S * S + s * S + tid;
        shared_data[tid] = scores[offset];
    }

    __syncthreads();

    if (tid < S) {
        shared_data[tid] = exp(shared_data[tid]);
    }

    __syncthreads();

    for (int stride = 1; stride < S; stride *= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    scalar_t sum = shared_data[0];

    __syncthreads();

    if (tid < S) {
        int offset = b * H * S * S + h * S * S + s * S + tid;
        scores[offset] = shared_data[tid] / sum;
    }
}

std::tuple<torch::Tensor> apply_softmax(
    torch::Tensor scores) {
    const auto B = scores.size(0);
    const auto H = scores.size(1);
    const auto S = scores.size(2);

    const dim3 threads(S);
    const dim3 blocks(B * H * S);

    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "softmax_kernel", ([&] {
        softmax_kernel<scalar_t><<<blocks, threads, S * sizeof(scalar_t)>>>(
            scores.data_ptr<scalar_t>(),
            B, H, S);
    }));

    return std::make_tuple(scores);
}
"""

# Custom CUDA kernel for final output computation (scores * V)
output_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void compute_output(
    const scalar_t* __restrict__ scores,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ output,
    int B, int H, int S, int D) {
    const int tile_size = 32;
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int tile_idx = blockIdx.y;
    int num_tiles = (S + tile_size - 1) / tile_size;
    int tile_row = tile_idx / num_tiles;
    int tile_col = tile_idx % num_tiles;
    int row_start = tile_row * tile_size;
    int col_start = tile_col * tile_size;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_in_tile = ty;
    int col_in_tile = tx;
    int global_row = row_start + row_in_tile;
    int global_col = col_start + col_in_tile;

    __shared__ scalar_t shared_S[tile_size][tile_size];
    __shared__ scalar_t shared_V[tile_size][tile_size];

    scalar_t sum = 0.0;

    for (int k = 0; k < S; k += tile_size) {
        // Load scores tile
        int S_offset = b * H * S * S + h * S * S;
        int S_row = global_row;
        int S_col = k + col_in_tile;
        if (S_row < S && S_col < S) {
            shared_S[row_in_tile][col_in_tile] = scores[S_offset + S_row * S + S_col];
        } else {
            shared_S[row_in_tile][col_in_tile] = 0.0;
        }

        // Load V tile
        int V_offset = b * H * S * D + h * S * D;
        int V_row = k + row_in_tile;
        int V_col = global_col;
        if (V_row < S && V_col < D) {
            shared_V[row_in_tile][col_in_tile] = V[V_offset + V_row * D + V_col];
        } else {
            shared_V[row_in_tile][col_in_tile] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < tile_size; ++i) {
            sum += shared_S[row_in_tile][i] * shared_V[i][col_in_tile];
        }

        __syncthreads();
    }

    if (global_row < S && global_col < D) {
        int output_offset = b * H * S * D + h * S * D + global_row * D + global_col;
        output[output_offset] = sum;
    }
}

std::tuple<torch::Tensor> compute_final_output(
    torch::Tensor scores,
    torch::Tensor V) {
    const auto B = scores.size(0);
    const auto H = scores.size(1);
    const auto S = scores.size(2);
    const auto D = V.size(3);

    auto output = torch::zeros({B, H, S, D}, V.options());

    const int tile_size = 32;
    const int num_tiles = (S + tile_size - 1) / tile_size;
    const dim3 threads(tile_size, tile_size);
    const dim3 blocks(B * H, num_tiles * num_tiles);

    AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "compute_output", ([&] {
        compute_output<scalar_t><<<blocks, threads>>>(
            scores.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, H, S, D);
    }));

    return std::make_tuple(output);
}
"""

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Compile attention_scores kernel
        self.attention_scores = load_inline(
            name="attention_scores",
            cuda_sources=attention_scores_source,
            functions=["attention_scores"],
            verbose=True
        )
        
        # Compile softmax kernel
        self.softmax = load_inline(
            name="softmax",
            cuda_sources=softmax_source,
            functions=["apply_softmax"],
            verbose=True
        )
        
        # Compile output kernel
        self.output = load_inline(
            name="output",
            cuda_sources=output_source,
            functions=["compute_final_output"],
            verbose=True
        )

    def forward(self, Q, K, V):
        scores = self.attention_scores.attention_scores(Q, K)
        scores = self.softmax.apply_softmax(scores)
        output = self.output.compute_final_output(scores, V)
        return output