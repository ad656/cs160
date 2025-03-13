#define TILE_WIDTH 16
#define KERNEL_SZ 7

// Default implementation
__kernel void im2col(__global float *unrolled, __global float *x, const int B,
                     const int C_in, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0)                                                    \
  x[(i3) * (C_in * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  // `unrolled` is a (B, H_unroll, W_unroll) tensor
#define x_unroll_3d(i2, i1, i0)                                                \
  unrolled[((i2) * (C_in * K * K) + (i1)) * ((H - K + 1) * (W - K + 1)) + (i0)]

  // Get thread indices
  int b = get_global_id(0);  // Batch index
  int c_in = get_global_id(1);  // Channel index
  int row_i = get_global_id(2);  // Input row index

  // Output dimensions
  int H_out = H - K + 1;
  int W_out = W - K + 1;

  // Iterate over input columns
  for (int col_i = 0; col_i < W; col_i++) {
    // Iterate over kernel rows and columns
    for (int mask_offset_row = 0; mask_offset_row < K; mask_offset_row++) {
      for (int mask_offset_col = 0; mask_offset_col < K; mask_offset_col++) {
        // Compute output position (row_o, col_o)
        int row_o = row_i - mask_offset_row;
        int col_o = col_i - mask_offset_col;

        // Check if output position is within bounds
        bool row_o_in_bounds = (row_o >= 0) && (row_o < H_out);
        bool col_o_in_bounds = (col_o >= 0) && (col_o < W_out);

        if (row_o_in_bounds && col_o_in_bounds) {
          // Compute indices in unrolled matrix
          int row_u = c_in * K * K + mask_offset_row * K + mask_offset_col;
          int col_u = row_o * W_out + col_o;

          // Copy value from input tensor to unrolled matrix
          x_unroll_3d(b, row_u, col_u) = x4d(b, c_in, row_i, col_i);
        }
      }
    }
  }

#undef x4d
#undef x_unroll_3d
}
