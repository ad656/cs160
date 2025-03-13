#include <cmath>
#include <iostream>
#include <vector>
#define TILE_WIDTH 16
#include <clblast.h>

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
    size_t x_size = sizeof(float) * B * C * H * W;
    size_t y_size = sizeof(float) * B * M * (H - K + 1) * (W - K + 1);
    size_t k_size = sizeof(float) * M * C * K * K;
    size_t x_unroll_size = sizeof(float) * B * C * K * K * (H - K + 1) * (W - K + 1);

    
    CHECK_ERR(err, "clCreateProgramWithSource");
    *device_x = clCreateBuffer(opencl ->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_size, (void *)host_x, &err);
    CHECK_ERR(err, "clCreateBuffer device_x");

    *device_k = clCreateBuffer(opencl ->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_size, (void *)host_k, &err);
    CHECK_ERR(err, "clCreateBuffer device_k");

    *device_y = clCreateBuffer(opencl ->context, CL_MEM_WRITE_ONLY, y_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_y");

    *device_x_unroll = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, x_unroll_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_x_unroll");
}


void OpenCLInterface::conv_forward_gemm_opencl(
    cl_mem device_y, const cl_mem device_x, const cl_mem device_k, 
    const cl_mem device_x_unroll, const int B, const int M, 
    const int C, const int H, const int W, const int K) 

    
{
    
    cl_int err;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int unrolled_width = H_out * W_out;
    const int unrolled_height = C * K * K;

    // Define global and local work sizes
    size_t global_size_im2col[3] = {(size_t)B, (size_t)C, (size_t)(H * W)};
    size_t local_size_im2col[3] = {1, 1, TILE_WIDTH}; // TILE_WIDTH = 16

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
    err = clEnqueueNDRangeKernel(opencl ->queue, opencl->im2col_kernel, 3, NULL, global_size_im2col, local_size_im2col, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel im2col");

    // GEMM operation using clBLAST
    float alpha = 1.0f;
    float beta = 0.0f;

    clblast::GemmBatched<float>(
        clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
        M, unrolled_width, unrolled_height,
        &alpha,  // ✅ Pass a float pointer instead of double
        device_k, 0, unrolled_height,
        device_x_unroll, 0, unrolled_width,
        &beta,  // ✅ Pass a float pointer instead of double
        device_y, 0, unrolled_width,
        B, &opencl->queue, nullptr);
    }

void OpenCLInterface::conv_forward_gemm_opencl_epilog(
    float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, 
    cl_mem device_x_unroll, const int B, const int M, 
    const int C, const int H, const int W, const int K) 
{
    cl_int err;
    size_t y_size = sizeof(float) * B * M * (H - K + 1) * (W - K + 1);
    err = clEnqueueReadBuffer(opencl ->queue, device_y, CL_TRUE, 0, y_size, host_y, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device_y");

    clReleaseMemObject(device_x);
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_k);
    clReleaseMemObject(device_x_unroll);
}
