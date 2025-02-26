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
    size_t size_y = B * M * H * W * sizeof(float);
    size_t size_x = B * C * H * W * sizeof(float);
    size_t size_k = M * C * K * K * sizeof(float);
    //@@ Allocate OpenCL memory here
    *device_y = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE, size_y, nullptr, &err);
    CHECK_ERR(err, "clCreateBuffer for device_y");
    
    *device_x = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY, size_x, nullptr, &err);
    CHECK_ERR(err, "clCreateBuffer for device_x");
    
    *device_k = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY, size_k, nullptr, &err);
    CHECK_ERR(err, "clCreateBuffer for device_k");
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
    err = clEnqueueWriteBuffer(this->opencl->queue, *device_x, CL_TRUE, 0, size_x, host_x, 0, nullptr, nullptr);
    CHECK_ERR(err, "clEnqueueWriteBuffer for device_x");
    
    err = clEnqueueWriteBuffer(this->opencl->queue, *device_k, CL_TRUE, 0, size_k, host_k, 0, nullptr, nullptr);
    CHECK_ERR(err, "clEnqueueWriteBuffer for device_k");

}


void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    size_t size_y = B * M * H * W * sizeof(float);
    //__global float *y, __constant float *x, __constant float *k,
    // const int B, const int M, const int C, const int H, const int W, const int K)
    // Set the arguments to our compute kernel
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Set the kernel dimensions and call the kernel
    err = clSetKernelArg(this->opencl->kernel, 0, sizeof(cl_mem), &device_y);
    CHECK_ERR(err, "clSetKernelArg for device_y");

    err = clSetKernelArg(this->opencl->kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg for device_x");

    err = clSetKernelArg(this->opencl->kernel, 2, sizeof(cl_mem), &device_k);
    CHECK_ERR(err, "clSetKernelArg for device_k");

    err = clSetKernelArg(this->opencl->kernel, 3, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg for B");

    err = clSetKernelArg(this->opencl->kernel, 4, sizeof(int), &M);
    CHECK_ERR(err, "clSetKernelArg for M");

    err = clSetKernelArg(this->opencl->kernel, 5, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg for C");

    err = clSetKernelArg(this->opencl->kernel, 6, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg for H");

    err = clSetKernelArg(this->opencl->kernel, 7, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg for B");

    err = clSetKernelArg(this->opencl->kernel, 8, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg for K");


    size_t globalWorkSize[3] = {static_cast<size_t>(B), static_cast<size_t>(M), static_cast<size_t>(H * W)};
    size_t localWorkSize[3] = {1, 1, TILE_WIDTH};

    //@@ Launch the OpenCL Kernel here
    // Execute the OpenCL kernel on the array

    err = clEnqueueNDRangeKernel(this->opencl->queue, this->opencl->kernel, 3, nullptr,globalWorkSize , localWorkSize, 0, nullptr, nullptr);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
}


void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    size_t size_y = B * M * H * W * sizeof(float);
    //@@ Copy the output back to host

    err = clEnqueueReadBuffer(this->opencl->queue, device_y, CL_TRUE, 0, size_y, host_y, 0, nullptr, nullptr);
    CHECK_ERR(err, "clEnqueueReadBuffer for device_y");
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
