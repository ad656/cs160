#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void conv_forward_kernel(__global float *y, __constant float *x, __constant float *k, 
                               const int B, const int M, const int C, const int H, const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // Get global thread indices
    const int m = get_global_id(0);  // Output feature map
    const int h_w = get_global_id(1); // Combined height/width position
    const int b = get_global_id(2);   // Batch
    
    // Ensure we're within bounds
    if (m >= M || b >= B)
        return;
    
    // Convert h_w to h and w
    const int W_grid = ceil(W_out * 1.0 / TILE_WIDTH);
    const int h = (h_w / W_grid) * TILE_WIDTH + get_local_id(1) / TILE_WIDTH;
    const int w = (h_w % W_grid) * TILE_WIDTH + get_local_id(1) % TILE_WIDTH;
    
    if (h >= H_out || w >= W_out)
        return;
    
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    // Compute single output element
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                sum += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
            }
        }
    }
    y4d(b, m, h, w) = sum;
    
#undef y4d
#undef x4d
#undef k4d
}
