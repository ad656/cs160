#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void conv_forward_kernel(__global float *y, __global float *x, __constant float *k, 
                                  const int B, const int M, const int C, const int H, const int W, const int K) {

     const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int w = get_global_id(0);
    const int h = get_global_id(1);
    const int m = get_global_id(2) % M;
    const int b = get_global_id(2) / M;

    if (w >= W_out || h >= H_out || m >= M || b >= B)
        return;

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your CPU convolution kernel code here

    for (int b = 0; b < B; b++) {
      for (int m = 0; m < M; m++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                y4d(b, m, h, w) = 0;
                for (int c = 0; c < C; c++) {
                    for (int p = 0; p < K; p++) {
                      for (int q = 0; q < K; q++) {
                        y4d(b, m, h, w) += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                      }
                    }
                }
            }
        }
      }
    }
#undef y4d
#undef x4d
#undef k4d

}
