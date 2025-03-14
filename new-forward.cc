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
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    size_t x_size = sizeof(float) * B * C * H * W; // input_size * sizeoffloat
    size_t y_size = sizeof(float) * B * M * H_out * W_out; 
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

    //clEnqueueWriteBuffer(opencl->queue, *device_x, CL_TRUE, 0, x_size, host_x, 0, NULL, NULL);

    //clEnqueueWriteBuffer(opencl->queue, *device_k, CL_TRUE, 0, k_size, host_k, 0, NULL, NULL);
}

void OpenCLInterface::conv_forward_gemm_opencl(
    cl_mem device_y, const cl_mem device_x, const cl_mem device_k, 
    const cl_mem device_x_unroll, const int B, const int M, 
    const int C, const int H, const int W, const int K) 
{
    cl_int err;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    
    size_t global_size_im2col[3] = {
        ((size_t)W_out + 15) / 16 * 16,  // Round up to multiple of 16
        ((size_t)H_out + 15) / 16 * 16,  // Round up to multiple of 16
        (size_t)(B*C)
    };
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

    // GEMM operation using clBLAST
    auto alpha = std::vector<float>(B,1.0f);  // Initialize with B elements set to 1.0
    auto beta = std::vector<float>(B,0.0f);   // Initialize with B elements set to 0.0


    std::vector<size_t> a_offsets(B), b_offsets(B), c_offsets(B);
    for (int i = 0; i < B; i++) {
        a_offsets[i] = i * M * (C * K * K);
        b_offsets[i] = i * (C * K * K) * (H_out * W_out);
        c_offsets[i] = i * M * (H_out * W_out);
    }
    //GEMM = alpha * A * B + beta * C
    //a = m * c * k * k
    //b = 
    clblast::GemmBatched(
        clblast::Layout::kRowMajor, 
        clblast::Transpose::kNo, 
        clblast::Transpose::kNo,
        M, 
        (W-K+1)*(H-K+1), 
        C * K * K,
        alpha.data(),
        device_k,
        a_offsets.data(), 
        C * K * K,
        device_x_unroll, 
        b_offsets.data(), 
        H_out * W_out,
        beta.data(),
        device_y, 
        c_offsets.data(), 
        H_out * W_out,
        B, 
        &opencl->queue, 
        nullptr);
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

    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x_unroll);
}
