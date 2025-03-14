#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void im2col(__global float *unrolled, __global float *x, 
    const int B, const int C_in, const int H, const int W, const int K) {
// Get thread indices - these correspond to output positions and channel/batch
const int col_o = get_global_id(0);    // Column index in output
const int row_o = get_global_id(1);    // Row index in output
const int bc_idx = get_global_id(2);   // Batch and channel index
const int b = bc_idx / C_in;           // Batch index
const int c = bc_idx % C_in;           // Channel index

// Calculate output dimensions
const int H_out = H - K + 1;
const int W_out = W - K + 1;

// Only process valid output positions
if (b < B && c < C_in && row_o < H_out && col_o < W_out) {
// For each position in the filter/kernel
for (int kh = 0; kh < K; kh++) {
for (int kw = 0; kw < K; kw++) {
// Calculate corresponding input position
int row_i = row_o + kh;
int col_i = col_o + kw;

// Calculate feature map index in the input tensor
int input_idx = b * (C_in * H * W) + c * (H * W) + row_i * W + col_i;

// Calculate unrolled matrix position
// Each column of unrolled corresponds to one output position (row_o, col_o)
// Each row corresponds to one input value from the receptive field (across all channels)
int filter_idx = (c * K * K) + (kh * K) + kw;
int output_pos = row_o * W_out + col_o;

// Final unrolled index calculation
int unroll_idx = b * (C_in * K * K * H_out * W_out) + 
               filter_idx * (H_out * W_out) + output_pos;

// Set the value in the unrolled matrix
unrolled[unroll_idx] = x[input_idx];
}
}
}
#undef x4d
#undef x_unroll_3d
}



