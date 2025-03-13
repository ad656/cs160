#include <cmath>
#include <iostream>
#include <vector>

#include <clblast.h>
#include "kernel.h"
#include "device.h"

#include "opencl-new-forward.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                            \
    if (err != CL_SUCCESS)                             \
    {                                                  \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                            \
    }

// Forward declaration of the im2col kernel
__kernel void im2col(__global float *x_unrolled, __global const float *x, 
                     const int B, const int C, const int H, const int W, 
                     const int K, const int H_out, const int W_out) {
    // Get thread indices
    const int b = get_global_id(0);  // Batch index
    const int h_out = get_global_id(1); // Output row
    const int w_out = get_global_id(2); // Output column
    
    // Output dimensions
    const int out_size = H_out * W_out;
    
    // Only compute if within bounds
    if (b < B && h_out < H_out && w_out < W_out) {
        // Calculate output position
        const int out_pos = h_out * W_out + w_out;
        
        // Iterate over input channels and kernel positions
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    // Calculate input position
                    const int h_in = h_out + p;
                    const int w_in = w_out + q;
                    
                    // Calculate indices
                    const int input_idx = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
                    const int unroll_idx = b * (C * K * K * out_size) + 
                                          (c * K * K + p * K + q) * out_size + 
                                          out_pos;
                    
                    // Copy data to unrolled matrix
                    x_unrolled[unroll_idx] = x[input_idx];
                }
            }
        }
    }
}

void OpenCLInterface::conv_forward_opencl_prolog(
    const float *host_y, const float *host_x, const float *host_k, 
    cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    
    // Calculate output dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // Calculate buffer sizes
    size_t x_size = B * C * H * W * sizeof(float);
    size_t k_size = M * C * K * K * sizeof(float);
    size_t y_size = B * M * H_out * W_out * sizeof(float);
    
    // Create memory buffers
    *device_x = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                               x_size, (void*)host_x, &err);
    CHECK_ERR(err, "clCreateBuffer device_x");
    
    *device_k = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                               k_size, (void*)host_k, &err);
    CHECK_ERR(err, "clCreateBuffer device_k");
    
    *device_y = clCreateBuffer(this->opencl->context, CL_MEM_WRITE_ONLY, 
                               y_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_y");
    
    // Create an additional buffer for unrolled input data
    size_t unroll_size = B * C * K * K * H_out * W_out * sizeof(float);
    this->device_x_unroll = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE, 
                                          unroll_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_x_unroll");
    
    // Create im2col kernel if not already created
    if (this->im2col_kernel == NULL) {
        // Create kernel
        this->im2col_kernel = clCreateKernel(this->opencl->program, "im2col", &err);
        CHECK_ERR(err, "clCreateKernel im2col");
    }
}

void OpenCLInterface::conv_forward_opencl(
    cl_mem device_y, const cl_mem device_x, const cl_mem device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    
    // Calculate output dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // Step 1: Perform im2col operation to unroll the input
    // Set kernel arguments for im2col
    err = clSetKernelArg(this->im2col_kernel, 0, sizeof(cl_mem), &this->device_x_unroll);
    CHECK_ERR(err, "clSetKernelArg x_unroll");
    err = clSetKernelArg(this->im2col_kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg x");
    err = clSetKernelArg(this->im2col_kernel, 2, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg B");
    err = clSetKernelArg(this->im2col_kernel, 3, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg C");
    err = clSetKernelArg(this->im2col_kernel, 4, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg H");
    err = clSetKernelArg(this->im2col_kernel, 5, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg W");
    err = clSetKernelArg(this->im2col_kernel, 6, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg K");
    err = clSetKernelArg(this->im2col_kernel, 7, sizeof(int), &H_out);
    CHECK_ERR(err, "clSetKernelArg H_out");
    err = clSetKernelArg(this->im2col_kernel, 8, sizeof(int), &W_out);
    CHECK_ERR(err, "clSetKernelArg W_out");
    
    // Launch im2col kernel
    size_t im2col_global_size[3] = {(size_t)B, (size_t)H_out, (size_t)W_out};
    size_t im2col_local_size[3] = {1, TILE_WIDTH, TILE_WIDTH};
    
    // Adjust local size if needed
    while (im2col_local_size[1] * im2col_local_size[2] > TILE_WIDTH * TILE_WIDTH) {
        if (im2col_local_size[1] > 1) im2col_local_size[1] /= 2;
        else if (im2col_local_size[2] > 1) im2col_local_size[2] /= 2;
    }
    
    err = clEnqueueNDRangeKernel(this->opencl->queue, this->im2col_kernel, 3, NULL, 
                                im2col_global_size, im2col_local_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel im2col");
    
    // Step 2: Perform batched matrix multiplication using clBLAST
    // For each batch, multiply the kernel weights with the unrolled input
    
    // GEMM parameters
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Calculate leading dimensions
    const int lda = C * K * K;  // Leading dimension of weights
    const int ldb = H_out * W_out;  // Leading dimension of unrolled input
    const int ldc = H_out * W_out;  // Leading dimension of output
    
    // Calculate offsets for each batch
    std::vector<size_t> a_offsets(B);
    std::vector<size_t> b_offsets(B);
    std::vector<size_t> c_offsets(B);
    
    for (int b = 0; b < B; b++) {
        // No offset for weights (device_k) as they are shared across all batches
        a_offsets[b] = 0;
        // Offset for unrolled input
        b_offsets[b] = b * (C * K * K * H_out * W_out);
        // Offset for output
        c_offsets[b] = b * (M * H_out * W_out);
    }
    
    // Perform batched GEMM
    auto status = clblast::GemmBatched<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo, clblast::Transpose::kNo,
        M, H_out * W_out, C * K * K,
        alpha,
        device_k, a_offsets.data(), lda,
        this->device_x_unroll, b_offsets.data(), ldb,
        beta,
        device_y, c_offsets.data(), ldc,
        B,
        &this->opencl->queue);
    
    if (status != clblast::StatusCode::kSuccess) {
        fprintf(stderr, "clBLAST GemmBatched failed: %d\n", static_cast<int>(status));
        exit(EXIT_FAILURE);
    }
}

void OpenCLInterface::conv_forward_opencl_epilog(
    float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    
    // Calculate output dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // Copy output back to host
    size_t y_size = B * M * H_out * W_out * sizeof(float);
    err = clEnqueueReadBuffer(this->opencl->queue, device_y, CL_TRUE, 0, y_size, host_y, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device_y");
    
    // Release memory
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
    clReleaseMemObject(this->device_x_unroll);
}
