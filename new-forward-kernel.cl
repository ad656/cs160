__kernel void conv_forward_kernel(__global float *y, __global float *x, __constant float *k, 
                              const int B, const int M, const int C, const int H, const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Get thread coordinates
    const int w_out = get_global_id(0);
    const int h_out = get_global_id(1);
    // Extract batch and feature map indices from third dimension
    const int batch_feature = get_global_id(2);
    const int m = batch_feature % M;
    const int b = batch_feature / M;

    // Boundary check
    if (w_out >= W_out || h_out >= H_out || m >= M || b >= B)
        return;

    // Define macros for indexing
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Compute output for this thread's pixel
    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += x4d(b, c, h_out+p, w_out+q) * k4d(m, c, p, q);
            }
        }
    }
    
    // Write result
    y4d(b, m, h_out, w_out) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
}
