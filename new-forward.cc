#include <cmath>
#include <iostream>
#include <vector>

#include <clblast.h>
#define TILE_WIDTH 16
#include "kernel.h"
#include "device.h"

#include "opencl-new-forward.h"

#define CHECK_ERR(err, msg)                            \
    if (err != CL_SUCCESS)                             \
    {                                                  \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                            \
    }
void OpenCLInterface::conv_forward_gemm_opencl_prolog(
    const float *host_y, const float *host_x, const float *host_k, 
    cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, cl_mem *device_x_unroll, 
    const int B, const int M, const int C, const int H, const int W, const int K) 
{
    cl_int err;
    
    size_t x_size = sizeof(float) * B * C * H * W; // input_size * sizeoffloat
    size_t y_size = sizeof(float) * B * M * (H - K + 1) * (W - K + 1); // Fixed output size calculation
    size_t k_size = sizeof(float) * M * C * K * K;
    size_t x_unroll_size = sizeof(float) * B * C * K * K * (H - K + 1) * (W - K + 1);

    *device_x = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_size, (void *)host_x, &err);
    CHECK_ERR(err, "clCreateBuffer device_x");

    *device_k = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_size, (void *)host_k, &err);
    CHECK_ERR(err, "clCreateBuffer device_k");

    *device_y = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY, y_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_y");

    *device_x_unroll = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, x_unroll_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_x_unroll");

    // No need to explicitly enqueue write buffer when using CL_MEM_COPY_HOST_PTR
}

void OpenCLInterface::conv_forward_gemm_opencl(
    cl_mem device_y, const cl_mem device_x, const cl_mem device_k, 
    const cl_mem device_x_unroll, const int B, const int M, 
    const int C, const int H, const int W, const int K) 
{
    cl_int err;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Calculate global and local work sizes for im2col
    size_t global_size_im2col[3] = {(size_t)W_out, (size_t)H_out, (size_t)(B*C)};
    size_t local_size_im2col[3] = {16, 16, 1}; // Adjusted for better performance
 
    // Set Kernel Arguments for im2col
    err = clSetKernelArg(opencl->im2col_kernel, 0, sizeof(cl_mem), &device_x_unroll);
    err |= clSetKernelArg(opencl->im2col_kernel, 1, sizeof(cl_mem), &device_x);
    err |= clSetKernelArg(opencl->im2col_kernel, 2, sizeof(int), &B);
    err |= clSetKernelArg(opencl->im2col_kernel, 3, sizeof(int), &C);
    err |= clSetKernelArg(opencl->im2col_kernel, 4, sizeof(int), &H);
    err |= clSetKernelArg(opencl->im2col_kernel, 5, sizeof(int), &W);
    err |= clSetKernelArg(opencl->im2col_kernel, 6, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg kernel_im2col");

    // Launch im2col kernel
    err = clEnqueueNDRangeKernel(opencl->queue, opencl->im2col_kernel, 3, NULL, global_size_im2col, local_size_im2col, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel im2col");
    clFinish(opencl->queue); // Ensure im2col completes before GEMM

    // For batch processing, we'll use a single GEMM call with batch handling
    // Each batch element is a separate GEMM operation
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // GEMM dimensions:
    // A (weights): M × (C*K*K)
    // B (unrolled input): (C*K*K) × (H_out*W_out)
    // C (output): M × (H_out*W_out)
    
    for (int b = 0; b < B; b++) {
        // Calculate offsets for this batch
        const size_t a_offset = 0; // weights are shared across batch
        const size_t b_offset = b * (C * K * K * H_out * W_out);
        const size_t c_offset = b * (M * H_out * W_out);
        
        // Use clBLAST's standard GEMM (not batched)
        const auto status = clblast::Gemm(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo,
            clblast::Transpose::kNo,
            M,                          // Number of rows in A and C
            H_out * W_out,              // Number of columns in B and C
            C * K * K,                  // Number of columns in A, rows in B
            alpha,                      // Alpha multiplier
            device_k,                   // Matrix A (weights)
            a_offset,
            C * K * K,                  // Leading dimension of A
            device_x_unroll,            // Matrix B (unrolled input)
            b_offset,
            H_out * W_out,              // Leading dimension of B
            beta,                       // Beta multiplier
            device_y,                   // Matrix C (output)
            c_offset,
            H_out * W_out,              // Leading dimension of C
            &opencl->queue,             // Command queue
            nullptr                     // Event
        );
        
        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "GEMM operation failed with error code: " << static_cast<int>(status) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    
    clFinish(opencl->queue); // Make sure all GEMM operations complete
}

void OpenCLInterface::conv_forward_gemm_opencl_epilog(
    float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, 
    cl_mem device_x_unroll, const int B, const int M, 
    const int C, const int H, const int W, const int K) 
{
    cl_int err;
    size_t y_size = sizeof(float) * B * M * (H - K + 1) * (W - K + 1);
    err = clEnqueueReadBuffer(opencl->queue, device_y, CL_TRUE, 0, y_size, host_y, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device_y");

    // Cleanup resources
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
    clReleaseMemObject(device_x_unroll);
}
