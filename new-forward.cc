#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"

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
    *device_x = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x_size, (void*)host_x, &err);
    CHECK_ERR(err, "clCreateBuffer x");

    size_t k_size = M * C * K * K * sizeof(float);
    *device_k = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k_size, (void*)host_k, &err);
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

    //__global float *y, __constant float *x, __constant float *k,
    // const int B, const int M, const int C, const int H, const int W, const int K)
    // Set the arguments to our compute kernel
    cl_int err;
    

    // Set kernel arguments
    err = clSetKernelArg(this->opencl->kernel, 0, sizeof(cl_mem), &device_y);
    CHECK_ERR(err, "clSetKernelArg y");
    err = clSetKernelArg(this->opencl->kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg x");
    err = clSetKernelArg(this->opencl->kernel, 2, sizeof(cl_mem), &device_k);
    CHECK_ERR(err, "clSetKernelArg k");
    err = clSetKernelArg(this->opencl->kernel, 3, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg B");
    err = clSetKernelArg(this->opencl->kernel, 4, sizeof(int), &M);
    CHECK_ERR(err, "clSetKernelArg M");
    err = clSetKernelArg(this->opencl->kernel, 5, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg C");
    err = clSetKernelArg(this->opencl->kernel, 6, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg H");
    err = clSetKernelArg(this->opencl->kernel, 7, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg W");
    err = clSetKernelArg(this->opencl->kernel, 8, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg K");

    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Set the kernel dimensions and call the kernel

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    
    int W_grid = ceil(W_out*1.0/TILE_WIDTH); 	
    int H_grid = ceil(H_out*1.0/TILE_WIDTH);

    int Y = H_grid * W_grid;
    // Define global and local work sizes
    size_t globalSize[3] = {(size_t)M, (size_t)Y, (size_t)B};
    size_t localSize[3] = { TILE_WIDTH, TILE_WIDTH, 1 };


    //@@ Launch the OpenCL Kernel here
    // Execute the OpenCL kernel on the array
    err = clEnqueueNDRangeKernel(this->opencl->queue, this->opencl->kernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
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
