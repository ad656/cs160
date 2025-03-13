#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"
#include <CL/cl.hpp>
#include <clblast.h>
#include "opencl-new-forward.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }
	
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

    // Reshape the input (im2col) and prepare matrices for GEMM
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int im2col_size = B * H_out * W_out * K * K * C;
    
    std::vector<float> im2col_matrix(im2col_size);  // Holds the reshaped input for GEMM
    std::vector<float> output_matrix(B * M * H_out * W_out, 0);  // Output matrix for GEMM result

    // Perform im2col: flatten each sliding window into a column of the matrix
    // This step will involve copying patches of the input tensor (device_x) into im2col_matrix

    // Call the corresponding CLBlast GemmBatched function for the matrix multiplication
    clblast::GemmBatched<float>(
        clblast::Layout::kRowMajor,    // Matrix layout (RowMajor)
        clblast::Transpose::kNo,       // No transpose for A
        clblast::Transpose::kNo,       // No transpose for B
        M * H_out * W_out,             // Rows of output matrix
        B,                            // Number of matrices in the batch
        K * K * C,                    // Columns of input and rows of kernel (im2col)
        1.0f,                         // Scalar alpha
        im2col_matrix.data(),         // Input matrix (im2col)
        K * K * C,                    // Leading dimension of im2col_matrix
        host_k,                       // Kernel matrix (flattened)
        K * K * C,                    // Leading dimension of kernel
        0.0f,                         // Scalar beta
        output_matrix.data(),         // Output matrix
        M * H_out * W_out,            // Leading dimension of output
        B,                            // Number of matrices in the batch
        this->opencl->queue()         // OpenCL queue for execution
    );

    // Copy the result back to the output buffer
    size_t y_size = B * M * H_out * W_out * sizeof(float);
    cl_int err = clEnqueueWriteBuffer(this->opencl->queue, device_y, CL_TRUE, 0, y_size, output_matrix.data(), 0, NULL, NULL);
    CHECK_ERR(err, "Copying output back to device_y");
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
