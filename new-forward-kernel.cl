#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void im2col(__global float *unrolled, __global float *x, 
                     const int B, const int C_in, const int H, const int W, 
                     const int K, const int H_unroll, const int W_unroll) {

    int b = get_global_id(0); // Batch index
    int c_in = get_global_id(1); // Channel index
    int index = get_global_id(2);
    int row_i = index / W;
    int col_i = index % W;

    if (b >= B || c_in >= C_in || row_i >= H || col_i >= W)
        return;

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int mask_offset_row = 0; mask_offset_row < K; ++mask_offset_row) {
        for (int mask_offset_col = 0; mask_offset_col < K; ++mask_offset_col) {
            int row_o = row_i - mask_offset_row;
            int col_o = col_i - mask_offset_col;
            bool row_o_in_bounds = (row_o >= 0 && row_o < H_out);
            bool col_o_in_bounds = (col_o >= 0 && col_o < W_out);
            
            if (row_o_in_bounds && col_o_in_bounds) {
                int col_u = row_o * W_out + col_o;
                int row_u = c_in * (K * K) + mask_offset_row * K + mask_offset_col;
                
                // Corrected indexing
                unrolled[(b * H_unroll * W_unroll) + (row_u * W_unroll) + col_u] = 
                    x[(b * C_in * H * W) + (c_in * H * W) + (row_i * W) + col_i];
            }
        }
    }
}
