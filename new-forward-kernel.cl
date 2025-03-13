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

#undef x4d
#undef x_unroll_3d
}
