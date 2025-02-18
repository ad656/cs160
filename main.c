#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <string.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include "img.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"

#define COMPUTE_OUTUT_DIM(input_dim, kernel_size, stride) \
    ((input_dim - kernel_size) / stride + 1)

void OpenCLConvolution2D(Image *input0, Matrix *input1, Image *result, int stride)
{
    // ... existing code ...

    //@@ Allocate GPU memory here
    device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                             sizeof(int) * input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS, 
                             NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_a");
    
    device_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
                             sizeof(float) * input1->shape[0] * input1->shape[1],
                             NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_b");
    
    device_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                             sizeof(int) * result->shape[0] * result->shape[1] * IMAGE_CHANNELS,
                             NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_c");

    //@@ Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0,
                              sizeof(int) * input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS,
                              input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_a");
    
    err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0,
                              sizeof(float) * input1->shape[0] * input1->shape[1],
                              input1->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_b");

    // ... existing code ...

    // @@ define local and global work sizes
    size_t localWorkSize[2] = {16, 16};  // Typical work group size
    size_t globalWorkSize[2] = {
        ((result->shape[1] + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0],
        ((result->shape[0] + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1]
    };
    
    //@@ Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");

    //@@ Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0,
                             sizeof(int) * result->shape[0] * result->shape[1] * IMAGE_CHANNELS,
                             result->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device_c");
    
    //@@ Free the GPU memory here
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernel_source);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // get the dir from the input file
    int stride;
    char dir[256];
    strcpy(dir, dirname(strdup(input_file_a))); 

    // Host input and output vectors and sizes
    Image host_a, host_c, answer;
    Matrix host_b;
    
    cl_int err;

    err = LoadImgRaw(input_file_a, &host_a);
    CHECK_ERR(err, "LoadImg");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    // err = LoadImgTmp(input_file_c, &answer);
    err = LoadImgRaw(input_file_c, &answer);
    CHECK_ERR(err, "LoadImg");

    // Load stride
    err = LoadStride(dir, &stride);
    CHECK_ERR(err, "LoadStride");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer image
    
    rows = COMPUTE_OUTUT_DIM(host_a.shape[0], host_b.shape[0], stride);
    cols = COMPUTE_OUTUT_DIM(host_a.shape[1], host_b.shape[0], stride);

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1] * IMAGE_CHANNELS);

    OpenCLConvolution2D(&host_a, &host_b, &host_c, stride);

    // Save the image
    SaveImg(input_file_d, &host_c);

    // Check the result of the convolution
    err = CheckImg(&answer, &host_c);
    CHECK_ERR(err, "CheckImg");

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}