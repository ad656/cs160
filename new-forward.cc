#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"
#include <CL/cl.hpp>
#include <clblast.h>
#include "opencl-new-forward.h"

#define TILE_WIDTH 16


	
void OpenCLInterface::conv_forward_opencl_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    //@@ Allocate OpenCL memory here

    size_t y_size = B * M * (H - K + 1) * (W - K + 1) * sizeof(float);
    *device_y = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE, y_size, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer y");

    size_t x_size = B * C * H * W * sizeof(float);
    *device_x = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_size, (void*)host_x, &err);
    CHECK_ERR(err, "clCreateBuffer x");

    size_t k_size = M * C * K * K * sizeof(float);
    *device_k = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_size, (void*)host_k, &err);
    CHECK_ERR(err, "clCreateBuffer k");
    // Create memory buffers for input and output vectors
    // 
    // Do not create your own device/context/queue. 
    // Use this->opencl->[program, kernel, queue, context]
    // OpenCL (common for entire NN)
    //      class is defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl.h
    //      methods defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6%2Fsrc%2Flayer%2Fcustom%opencl.cc
    //      created and passed into the network here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/m2.cc
    //      it's pointer is kept in OpenCLInterface (THIS) class here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl-new-forward.h

    //@@ Copy memory to the OpenCL here
    // Copy input vectors to memory buffers
    err = clEnqueueWriteBuffer(this->opencl->queue, *device_x, CL_TRUE, 0, x_size, host_x, 0, NULL, NULL);

    CHECK_ERR(err, "Copying host_x to device");

    err = clEnqueueWriteBuffer(this->opencl->queue, *device_k, CL_TRUE, 0, k_size, host_k, 0, NULL, NULL);

    CHECK_ERR(err, "Copying host_k to device");
}


void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    clblast::StatusCode err;

    // Calculate output dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Define matrix dimensions for GEMM
    size_t A_rows = B * C * H * W;   // Input flattened (B * C * H * W)
    size_t B_cols = M * C * K * K;   // Kernel flattened (M * C * K * K)
    size_t C_cols = B * M * H_out * W_out; // Output flattened (B * M * H_out * W_out)

    // Define leading dimensions
    size_t A_ld = A_rows;  // Leading dimension of input matrix (rows of A)
    size_t B_ld = M * K * K; // Leading dimension of kernel matrix (rows of B)
    size_t C_ld = C_cols;  // Leading dimension of output matrix (rows of C)

    // Prepare GEMM parameters
    float alpha = 1.0f;
    float beta = 0.0f;

    // Reshape input and kernel into the right format for GEMM (not shown, but this involves creating a flattened matrix for input and kernel)
    cl_mem reshaped_x = device_x;  // Assume reshaped_x buffer already created
    cl_mem reshaped_k = device_k;  // Assume reshaped_k buffer already created
    cl_mem reshaped_y = device_y;  // Output buffer

    // Batch count is B * M since each (B, M) pair is a separate matrix multiplication task
    size_t batch_count = B * M;

    // Perform GEMM using clBLAS::GemmBatched
    err = clblast::GemmBatched<float>(
        clblast::Layout::kRowMajor,  // Layout (RowMajor)
        clblast::Transpose::kNo,    // No transpose for input
        clblast::Transpose::kNo,    // No transpose for kernel
        A_rows,                     // Rows of A (input)
        C_cols,                     // Columns of C (output)
        B_cols,                     // Columns of B (kernel)
        &alpha,                     // Scalar alpha
        reshaped_x, NULL, A_ld,    // Input matrix A and leading dimension
        reshaped_k, NULL, B_ld,    // Kernel matrix B and leading dimension
        &beta,                      // Scalar beta
        reshaped_y, NULL, C_ld,    // Output matrix C and leading dimension
        batch_count,                // Batch count
        &this->opencl->queue,       // OpenCL command queue
        NULL                        // Event (optional, can be used to track completion)
    );
    CHECK_ERR(err, "clblast::GemmBatched failed");

    // Optionally, read back the results if necessary (not required for GEMM)
    // err = clEnqueueReadBuffer(this->opencl->queue, reshaped_y, CL_TRUE, 0, C_cols * sizeof(float), host_y, 0, NULL, NULL);
    // CHECK_ERR(err, "clEnqueueReadBuffer");

    // No need for kernel launch or argument setting anymore, since we're using GEMM
}


void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    //@@ Copy the output back to host
    size_t y_size = B * M * (H - K + 1) * (W - K + 1) * sizeof(float);
    err = clEnqueueReadBuffer(this->opencl->queue, device_y, CL_TRUE, 0, y_size, host_y, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer");

    // Read the memory buffer output_mem_obj to the local variable result
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Free the OpenCL memory here
    // Release OpenCL resources

    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
}
