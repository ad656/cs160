#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void do_not_remove_this_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void prefn_marker_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void conv_forward_kernel(
    __global float *y, 
    __global const float *x, 
    __constant float *k, 
    const int B, 
    const int M, 
    const int C, 
    const int H, 
    const int W, 
    const int K) 
{
    // Output dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // Local memory for input tile + halo regions
    __local float inputTile[TILE_WIDTH + KERNEL_SZ - 1][TILE_WIDTH + KERNEL_SZ - 1];
    
    // Thread indices
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    // Output indices
    const int w_out_base = get_group_id(0) * TILE_WIDTH;
    const int h_out_base = get_group_id(1) * TILE_WIDTH;
    const int batch_feature = get_global_id(2);
    const int m = batch_feature % M;
    const int b = batch_feature / M;
    
    // Macros for indexing
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    // Boundary check for batch and feature map
    if (b >= B || m >= M)
        return;
    
    // Process each input channel
    for (int c = 0; c < C; c++) {
        // Load input tile with halo regions into local memory
        for (int i = ty; i < TILE_WIDTH + K - 1; i += TILE_WIDTH) {
            for (int j = tx; j < TILE_WIDTH + K - 1; j += TILE_WIDTH) {
                int h_in = h_out_base + i;
                int w_in = w_out_base + j;
                
                if (h_in < H && w_in < W) {
                    inputTile[i][j] = x4d(b, c, h_in, w_in);
                } else {
                    inputTile[i][j] = 0.0f;
                }
            }
        }
        
        // Wait for all threads to finish loading the tile
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Each thread computes one output element
        int h_out = h_out_base + ty;
        int w_out = w_out_base + tx;
        
        // Only compute if within output bounds
        if (h_out < H_out && w_out < W_out) {
            // Compute convolution for this output element
            float acc = 0.0f;
            
            // If this is the first channel, initialize the accumulator
            // Otherwise add to the existing value
            if (c == 0) {
                acc = 0.0f;
            } else {
                acc = y4d(b, m, h_out, w_out);
            }
            
            // Compute convolution over the kernel
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += inputTile[ty + p][tx + q] * k4d(m, c, p, q);
                }
            }
            
            // Write result
            y4d(b, m, h_out, w_out) = acc;
        }
        
        // Wait for all threads to finish using the tile before loading the next one
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    #undef y4d
    #undef x4d
    #undef k4d
}
