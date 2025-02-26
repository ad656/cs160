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
        //@@ Allocate OpenCL memory here
        // Create memory buffers for input and output vectors
        // 
        // Do not create your own device/context/queue. 
        // Use this->opencl->[program, kernel, queue, context]
        
        cl_int err;
        
        // Calculate output dimensions
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
        
        // Allocate device memory
        *device_y = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY, 
                                  B * M * H_out * W_out * sizeof(float), NULL, &err);
        CHECK_ERR(err, "Creating device_y buffer");
        
        *device_x = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY, 
                                  B * C * H * W * sizeof(float), NULL, &err);
        CHECK_ERR(err, "Creating device_x buffer");
        
        *device_k = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY, 
                                  M * C * K * K * sizeof(float), NULL, &err);
        CHECK_ERR(err, "Creating device_k buffer");
        
        //@@ Copy memory to the OpenCL here
        // Copy input vectors to memory buffers
        
        err = clEnqueueWriteBuffer(opencl->queue, *device_x, CL_TRUE, 0,
                                  B * C * H * W * sizeof(float), host_x, 0, NULL, NULL);
        CHECK_ERR(err, "Copying host_x to device");
        
        err = clEnqueueWriteBuffer(opencl->queue, *device_k, CL_TRUE, 0,
                                  M * C * K * K * sizeof(float), host_k, 0, NULL, NULL);
        CHECK_ERR(err, "Copying host_k to device");
    }
    
    void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
    {
        // Set the arguments to our compute kernel
        //
        // Do not create your own device/context/queue.
        // Use this->opencl->[program, kernel, queue, context]
        
        cl_int err;
        cl_kernel conv_kernel = clCreateKernel(opencl->program, "conv_forward_kernel", &err);
        CHECK_ERR(err, "Creating kernel");
        
        // Calculate output dimensions
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
        
        // Set kernel arguments
        err = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &device_y);
        CHECK_ERR(err, "Setting kernel arg 0");
        
        err = clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &device_x);
        CHECK_ERR(err, "Setting kernel arg 1");
        
        err = clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &device_k);
        CHECK_ERR(err, "Setting kernel arg 2");
        
        err = clSetKernelArg(conv_kernel, 3, sizeof(int), &B);
        CHECK_ERR(err, "Setting kernel arg 3");
        
        err = clSetKernelArg(conv_kernel, 4, sizeof(int), &M);
        CHECK_ERR(err, "Setting kernel arg 4");
        
        err = clSetKernelArg(conv_kernel, 5, sizeof(int), &C);
        CHECK_ERR(err, "Setting kernel arg 5");
        
        err = clSetKernelArg(conv_kernel, 6, sizeof(int), &H);
        CHECK_ERR(err, "Setting kernel arg 6");
        
        err = clSetKernelArg(conv_kernel, 7, sizeof(int), &W);
        CHECK_ERR(err, "Setting kernel arg 7");
        
        err = clSetKernelArg(conv_kernel, 8, sizeof(int), &K);
        CHECK_ERR(err, "Setting kernel arg 8");
        
        //@@ Set the kernel dimensions and call the kernel
        //@@ Launch the OpenCL Kernel here
        // Execute the OpenCL kernel on the array
        
        // Define global and local work sizes
        size_t localWorkSize[3] = {TILE_WIDTH, TILE_WIDTH, 1};
        size_t globalWorkSize[3] = {
            ((W_out - 1) / TILE_WIDTH + 1) * TILE_WIDTH,
            ((H_out - 1) / TILE_WIDTH + 1) * TILE_WIDTH,
            (size_t)(B * M)
        };
        
        // Launch kernel
        err = clEnqueueNDRangeKernel(opencl->queue, conv_kernel, 3, NULL,
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
        CHECK_ERR(err, "Enqueueing kernel");
        
        // Wait for kernel to finish
        err = clFinish(opencl->queue);
        CHECK_ERR(err, "Waiting for kernel execution");
        
        // Release kernel
        err = clReleaseKernel(conv_kernel);
        CHECK_ERR(err, "Releasing kernel");
    }
    
    void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
    {
        //@@ Copy the output back to host
        // Read the memory buffer output_mem_obj to the local variable result
        //
        // Do not create your own device/context/queue.
        // Use this->opencl->[program, kernel, queue, context]
        
        cl_int err;
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
        
        // Copy the output back to host
        err = clEnqueueReadBuffer(opencl->queue, device_y, CL_TRUE, 0,
                                 B * M * H_out * W_out * sizeof(float), host_y, 0, NULL, NULL);
        CHECK_ERR(err, "Reading output buffer");
        
        //@@ Free the OpenCL memory here
        // Release OpenCL resources
        
        err = clReleaseMemObject(device_y);
        CHECK_ERR(err, "Releasing device_y");
        
        err = clReleaseMemObject(device_x);
        CHECK_ERR(err, "Releasing device_x");
        
        err = clReleaseMemObject(device_k);
        CHECK_ERR(err, "Releasing device_k");
    }
