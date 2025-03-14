#define TILE_WIDTH 16
#define KERNEL_SZ 7

// default implementation
__kernel void im2col(__global float *unrolled, __global float *x, const int B,
                     const int C_in, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0)                                                    \
  x[(i3) * (C_in * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  // `unrolled` is a (B, H_unroll, W_unroll) tensor
#define x_unroll_3d(i2, i1, i0)                                                \
  unrolled[((i2) * H_unroll + (i1)) * W_unroll + (i0)]

  //@@ Define your im2col operations here.

  int b = get_global_id(0);
  int c_in = get_global_id(1);
  int row_i = get_global_id(2) / W;
  int col_i = get_global_id(2) % W;

  if (b >= B || c_in >= C_in || row_i >= H || col_i >= W)
    return;

  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int H_unroll = C_in * K * K;
  int W_unroll = H_out * W_out;

  for (int mask_offset_row = 0; mask_offset_row < K; ++mask_offset_row) {
    for (int mask_offset_col = 0; mask_offset_col < K; ++mask_offset_col) {
      int row_o = row_i - mask_offset_row;
      int col_o = col_i - mask_offset_col;
      bool row_o_in_bounds = (row_o >= 0 && row_o < H_out);
      bool col_o_in_bounds = (col_o >= 0 && col_o < W_out);
      
      if (row_o_in_bounds && col_o_in_bounds) {
        int col_u = row_o * W_out + col_o;
        int row_u = c_in * (K * K) + mask_offset_row * K + mask_offset_col;
        
        x_unroll_3d(b, row_u, col_u) = x4d(b, c_in, row_i, col_i);
      }
    }
  }
#undef x4d
#undef x_unroll_3d
}