#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void conv_forward_kernel(__global float *y, __constant float *x, __constant float *k, 
                                  const int B, const int M, const int C, const int H, const int W, const int K) {

    int W_out = W - K + 1;
    int H_out = H - K + 1;

    int b = get_global_id(2);  // Batch index
    int m = get_global_id(0);  // Output channel (M)
    int h = get_global_id(1) / W_out; // Row index
    int w = get_global_id(1) % W_out; // Column index

    if (b < B && m < M && h < H_out && w < W_out) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    sum += x[(b * C * H * W) + (c * H * W) + ((h + p) * W) + (w + q)] * 
                           k[(m * C * K * K) + (c * K * K) + (p * K) + q];
                }
            }
        }
        y[(b * M * H_out * W_out) + (m * H_out * W_out) + (h * W_out) + w] = sum;
    }
}
