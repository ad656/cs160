#define TILE_WIDTH 16
#define KERNEL_SZ 7

// Default implementation
__kernel void im2col(__global float *unrolled, __global float *x, const int B,
    const int C_in, const int H, const int W, const int K) {

// Define macros for easy indexing
#define x4d(i3, i2, i1, i0) \
x[(i3) * (C_in * H * W) + (i2) * (H * W) + (i1) * (W) + (i0)]

// Define the unrolled dimensions
const int H_out = H - K + 1;
const int W_out = W - K + 1;
const int W_unroll = H_out * W_out;

// Define macro for unrolled indexing
#define x_unroll_3d(i2, i1, i0) \
unrolled[((i2) * (C_in * K * K) + (i1)) * W_unroll + (i0)]

// Get thread indices
const int col_i = get_global_id(0);    // Column index in input
const int row_i = get_global_id(1);    // Row index in input
const int bc_idx = get_global_id(2);   // Batch and channel index
const int b = bc_idx / C_in;           // Batch index
const int c_in = bc_idx % C_in;        // Channel index

// Only process if we're within bounds
if (b < B && c_in < C_in && row_i < H && col_i < W) {
// For each position in the filter
for (int mask_offset_row = 0; mask_offset_row < K; mask_offset_row++) {
for (int mask_offset_col = 0; mask_offset_col < K; mask_offset_col++) {
// Following the pseudocode: indices in the output where this input element contributes
// For each element in the input, we need to determine which output positions 
// it contributes to based on the filter position

// row_o = row_i - mask_offset_row
int row_o = row_i + mask_offset_row - (K-1);
int col_o = col_i + mask_offset_col - (K-1);

// Check if the output indices are within bounds
bool row_o_in_bounds = (0 <= row_o && row_o < H_out);
bool col_o_in_bounds = (0 <= col_o && col_o < W_out);

if (row_o_in_bounds && col_o_in_bounds) {
   // Calculate the unrolled matrix indices
   // row_u = filter position in the unrolled matrix
   int row_u = c_in * K * K + mask_offset_row * K + mask_offset_col;
   
   // col_u = output position in the unrolled matrix
   int col_u = row_o * W_out + col_o;
   
   // Set the value in the unrolled matrix
   // Unrolled is a 3D tensor: (batch, C*K*K, H_out*W_out)
   int unroll_idx = b * (C_in * K * K * W_unroll) + row_u * W_unroll + col_u;
   
   // Set the value from the input
   unrolled[unroll_idx] = x4d(b, c_in, row_i, col_i);
}
}
}
}

#undef x4d
#undef x_unroll_3d
}
