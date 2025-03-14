#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void im2col(__global float *unrolled, __global float *x, const int B,
                     const int C_in, const int H, const int W, const int K) {
    // Define macros for easy indexing
#define x4d(i3, i2, i1, i0) \
    x[(i3) * (C_in * H * W) + (i2) * (H * W) + (i1) * (W) + (i0)]

    // Define the unrolled dimensions
    const int H_out = H - K + 1; // Output height after convolution
    const int W_out = W - K + 1; // Output width after convolution
    const int H_unroll = C_in * K * K; // Number of rows in the unrolled matrix
    const int W_unroll = H_out * W_out; // Number of columns in the unrolled matrix

    // Define macro for unrolled indexing
#define x_unroll_3d(i2, i1, i0) \
    unrolled[((i2) * H_unroll + (i1)) * W_unroll + (i0)]

    // Get thread indices
    const int col_i = get_global_id(0);    // Column index in input
    const int row_i = get_global_id(1);    // Row index in input
    const int bc_idx = get_global_id(2);   // Batch and channel index
    const int b = bc_idx / C_in;           // Batch index
    const int c_in = bc_idx % C_in;        // Channel index

    // Only process if we're within bounds
    if (b < B && c_in < C_in && row_i < H && col_i < W) {
        // For each position in the filter/kernel
        for (int mask_offset_row = 0; mask_offset_row < K; mask_offset_row++) {
            for (int mask_offset_col = 0; mask_offset_col < K; mask_offset_col++) {
                // Calculate output position
                int row_o = row_i - mask_offset_row;
                int col_o = col_i - mask_offset_col;

                // Check if the output indices are within bounds
                bool row_o_in_bounds = (0 <= row_o && row_o < H_out);
                bool col_o_in_bounds = (0 <= col_o && col_o < W_out);

                if (row_o_in_bounds && col_o_in_bounds) {
                    // Calculate the unrolled matrix indices
                    // Each column corresponds to one output position (row_o, col_o)
                    int col_u = row_o * W_out + col_o;

                    // Each row corresponds to one input value from the receptive field
                    // Flattened in channel-major order, then row-major order
                    int row_u = c_in * K * K + mask_offset_row * K + mask_offset_col;

                    // Set the value in the unrolled matrix
                    x_unroll_3d(b, row_u, col_u) = x4d(b, c_in, row_i, col_i);
                }
            }
        }
    }

#undef x4d
#undef x_unroll_3d
}
