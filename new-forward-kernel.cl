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

__kernel void conv_forward_kernel(__global float *y, __constant float *x, __constant float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    int W_out = W - K + 1;
    int H_out = H - K + 1;
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    
int W_grid = ceil(W_out*1.0/TILE_WIDTH); 	// number of horizontal tiles per output map

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    
    float acc = 0.;
    if (h < H_out && w < W_out) {
        for (int c = 0;  c < C; c++) {		// sum over all input channels
            for (int p = 0; p < K; p++) {		// loop over KxK  filter
                for (int q = 0; q < K; q++)  {
                    acc += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = acc;
    } else {
        return;
    }

#undef y4d
#undef x4d
#undef k4d
}
