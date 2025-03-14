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
    
    // Calculate correct output dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    size_t x_size = sizeof(float) * B * C * H * W;
    size_t y_size = sizeof(float) * B * M * H_out * W_out; // Fixed output size calculation
    size_t k_size = sizeof(float) * M * C * K * K;
    size_t x_unroll_size = sizeof(float) * B * C * K * K * H_out * W_out;

    *device_x = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_size, (void *)host_x, &err);
    CHECK_ERR(err, "clCreateBuffer device_x");

    *device_k = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_size, (void *)host_k, &err);
    CHECK_ERR(err, "clCreateBuffer device_k");

    *device_y = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY, y_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_y");

    *device_x_unroll = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, x_unroll_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_x_unroll");

    // No need for explicit write buffer as we use CL_MEM_COPY_HOST_PTR
}

void OpenCLInterface::conv_forward_gemm_opencl(
    cl_mem device_y, const cl_mem device_x, const cl_mem device_k, 
    const cl_mem device_x_unroll, const int B, const int M, 
    const int C, const int H, const int W, const int K) 
{
    cl_int err;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // Configure global work sizes for im2col kernel
    // Use W and H for input dimensions and B*C for batch and channels
    size_t global_size_im2col[3] = {(size_t)W, (size_t)H, (size_t)(B*C)};
    size_t local_size_im2col[3] = {16, 16, 1};
 
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
    
    // Make sure im2col is complete before proceeding
    clFinish(opencl->queue);

    // Prepare for GEMM operation using clBLAST
    std::vector<float> alpha(B, 1.0f);
    std::vector<float> beta(B, 0.0f);

    // Perform GemmBatched operation
    // For each batch, we need to compute a separate GEMM operation
    // where weights (M x C*K*K) multiply unrolled input (C*K*K x H_out*W_out)
    
    // Calculate batch offsets correctly
    std::vector<size_t> a_offsets(B), b_offsets(B), c_offsets(B);
    for (int i = 0; i < B; i++) {
        // The weight matrix is shared across all batches, so offset is 0
        a_offsets[i] = 0;
        // Each batch has its own unrolled input starting at this offset
        b_offsets[i] = i * (C * K * K * H_out * W_out);
        // Each batch has its own output starting at this offset
        c_offsets[i] = i * (M * H_out * W_out);
    }

    // Execute the batched GEMM operation
    auto status = clblast::GemmBatched(
        clblast::Layout::kRowMajor,    // Use row-major layout
        clblast::Transpose::kNo,       // Don't transpose A
        clblast::Transpose::kNo,       // Don't transpose B
        M,                             // Number of rows in A and C
        H_out * W_out,                 // Number of columns in B and C
        C * K * K,                     // Number of columns in A, rows in B
        alpha.data(),                  // Alpha values for each batch
        device_k,                      // Matrix A (weights)
        a_offsets.data(),              // Offsets for A
        C * K * K,                     // Leading dimension of A
        device_x_unroll,               // Matrix B (unrolled input)
        b_offsets.data(),              // Offsets for B
        H_out * W_out,                 // Leading dimension of B
        beta.data(),                   // Beta values for each batch
        device_y,                      // Matrix C (output)
        c_offsets.data(),              // Offsets for C
        H_out * W_out,                 // Leading dimension of C
        B,                             // Number of batches
        &opencl->queue,                // Command queue
        nullptr                        // Event
    );

    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "GEMM operation failed with error code: " << static_cast<int>(status) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Ensure all operations complete
    clFinish(opencl->queue);
}

void OpenCLInterface::conv_forward_gemm_opencl_epilog(
    float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, 
    cl_mem device_x_unroll, const int B, const int M, 
    const int C, const int H, const int W, const int K) 
{
    cl_int err;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    size_t y_size = sizeof(float) * B * M * H_out * W_out;
    
    // Read back the results
    err = clEnqueueReadBuffer(opencl->queue, device_y, CL_TRUE, 0, y_size, host_y, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device_y");

    // Clean up resources
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
    clReleaseMemObject(device_x_unroll);
}
