#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void conv_forward_kernel(__global float *y, __constant float *x, __constant float *k, 
                                   const int B, const int M, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Define macros for easy access
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Calculate the grid dimensions
    int W_grid = (int) ceil(W_out * 1.0 / TILE_WIDTH);  // Number of horizontal tiles per output map
    
    // Get work-item IDs
    int b = get_global_id(2);  // Batch index
    int m = get_global_id(0);  // Output channel (filter) index
    int h = (get_global_id(1) / W_grid) * TILE_WIDTH + get_local_id(1);  // Output height index
    int w = (get_global_id(1) % W_grid) * TILE_WIDTH + get_local_id(0);  // Output width index

    float acc = 0.0f;
    
    // Compute the convolution only if within bounds
    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {  // Sum over all input channels
            for (int p = 0; p < K; p++) {  // Loop over KxK filter
                for (int q = 0; q < K; q++)  {
                    acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                }
            }
        }
        // Write the result to the output tensor
        y4d(b, m, h, w) = acc;
    }

    // Cleanup macros
    #undef y4d
    #undef x4d
    #undef k4d
}
