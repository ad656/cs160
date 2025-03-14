#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void im2col(__global float *unrolled, __global float *x, const int B,
                     const int C_in, const int H, const int W, const int K) {
    // Define the unrolled dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // Get thread indices - now we iterate over output positions
    const int col_o = get_global_id(0);    // Column index in output
    const int row_o = get_global_id(1);    // Row index in output
    const int bc_idx = get_global_id(2);   // Batch and channel index
    const int b = bc_idx / C_in;           // Batch index
    const int c_in = bc_idx % C_in;        // Channel index
    
    // Only process if we're within bounds
    if (b < B && c_in < C_in && row_o < H_out && col_o < W_out) {
        // Calculate position in unrolled matrix
        int col_u = row_o * W_out + col_o;  // Output position
        
        // For each position in the filter/kernel
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                // Calculate input position
                int row_i = row_o + kh;
                int col_i = col_o + kw;
                
                // Calculate row in unrolled matrix
                int row_u = c_in * K * K + kh * K + kw;
                
                // Set the value in the unrolled matrix
                unrolled[b * (C_in * K * K * H_out * W_out) + row_u * (H_out * W_out) + col_u] = 
                    x[b * (C_in * H * W) + c_in * (H * W) + row_i * W + col_i];
            }
        }
    }
}
