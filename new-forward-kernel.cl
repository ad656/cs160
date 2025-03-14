#define TILE_WIDTH 16
#define KERNEL_SZ 7

___kernel void im2col(__global float *unrolled, __global float *x, const int B,
    const int C_in, const int H, const int W, const int K) {
// Get output coordinates
const int col_o = get_global_id(0);  // Column in output
const int row_o = get_global_id(1);  // Row in output
const int bc_idx = get_global_id(2);
const int b = bc_idx / C_in;
const int c_in = bc_idx % C_in;

const int H_out = H - K + 1;
const int W_out = W - K + 1;

// Check bounds
if (row_o < H_out && col_o < W_out && b < B && c_in < C_in) {
// Calculate output column index
int col_u = row_o * W_out + col_o;

// For each position in kernel
for (int kh = 0; kh < K; kh++) {
for (int kw = 0; kw < K; kw++) {
// Calculate input position
int row_i = row_o + kh;
int col_i = col_o + kw;

// Calculate unrolled row index
int row_u = c_in * K * K + kh * K + kw;

// Set value
int unroll_idx = b * (C_in * K * K * H_out * W_out) + row_u * (H_out * W_out) + col_u;
int input_idx = b * (C_in * H * W) + c_in * (H * W) + row_i * W + col_i;
unrolled[unroll_idx] = x[input_idx];
}
}
}
}
