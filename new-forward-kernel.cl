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
    const int col_o = get_global_id(0);    // Column index in output
    const int row_o = get_global_id(1);    // Row index in output
    const int bc_idx = get_global_id(2);   // Batch and channel index
    const int b = bc_idx / C_in;           // Batch index
    const int c_in = bc_idx % C_in;        // Channel index

    // Only process if we're within bounds
    if (b < B && c_in < C_in && row_o < H_out && col_o < W_out) {
        // For each position in the filter/kernel
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                // Calculate corresponding input position
                int row_i = row_o + kh;
                int col_i = col_o + kw;

                // Check if the input position is within bounds
                if (row_i >= 0 && row_i < H && col_i >= 0 && col_i < W) {
                    // Calculate feature map index in the input tensor
                    int input_idx = b * (C_in * H * W) + c_in * (H * W) + row_i * W + col_i;

                    // Calculate unrolled matrix position
                    // Each column corresponds to one output position (row_o, col_o)
                    int col_u = row_o * W_out + col_o;

                    // Each row corresponds to one input value from the receptive field
                    // Flattened in channel-major order, then row-major order
                    int row_u = c_in * K * K + kh * K + kw;

                    // Set the value in the unrolled matrix
                    x_unroll_3d(b, row_u, col_u) = x4d(b, c_in, row_i, col_i);
                }
            }
        }
    }

#undef x4d
#undef x_unroll_3d
}
