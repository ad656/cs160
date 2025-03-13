#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <string.h>
#include <CL/cl.h>
#include <clblast.h>

#include "device.h"
#include "matrix.h"
#include "img.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define COMPUTE_OUTPUT_DIM(input_dim, kernel_size, stride) \
    ((input_dim - kernel_size) / stride + 1)

void CLBlastConvolution2D(Image *input0, Matrix *input1, Image *result, int stride)
{
    cl_int err;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;

    // Get OpenCL device and context
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Allocate OpenCL memory buffers
    cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                     sizeof(float) * input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS, 
                                     NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_a");

    cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                     sizeof(float) * input1->shape[0] * input1->shape[1], 
                                     NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_b");

    cl_mem device_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                     sizeof(float) * result->shape[0] * result->shape[1] * IMAGE_CHANNELS, 
                                     NULL, &err);
    CHECK_ERR(err, "clCreateBuffer device_c");

    // Copy data to GPU
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0,
                               sizeof(float) * input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS,
                               input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_a");

    err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0,
                               sizeof(float) * input1->shape[0] * input1->shape[1],
                               input1->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_b");

    // Perform batched GEMM using CLBlast
    const size_t batch_count = IMAGE_CHANNELS;
    const size_t m = result->shape[0] * result->shape[1];
    const size_t n = 1;
    const size_t k = input1->shape[0] * input1->shape[1];
    const float alpha = 1.0f;
    const float beta = 0.0f;

    err = CLBlastSgemmBatched(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                              m, n, k, alpha,
                              device_a, 0, k,
                              device_b, 0, n,
                              beta,
                              device_c, 0, n,
                              batch_count,
                              &queue, NULL);
    CHECK_ERR(err, "CLBlastSgemmBatched");

    // Copy result back to host
    err = clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0,
                              sizeof(float) * result->shape[0] * result->shape[1] * IMAGE_CHANNELS,
                              result->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device_c");

    // Cleanup
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
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
    const char *output_file = argv[4];

    char dir[256];
    strcpy(dir, dirname(strdup(input_file_a)));

    int stride;
    LoadStride(dir, &stride);

    Image host_a, host_c, answer;
    Matrix host_b;

    LoadImgRaw(input_file_a, &host_a);
    LoadMatrix(input_file_b, &host_b);
    LoadImgRaw(input_file_c, &answer);

    int rows = COMPUTE_OUTPUT_DIM(host_a.shape[0], host_b.shape[0], stride);
    int cols = COMPUTE_OUTPUT_DIM(host_a.shape[1], host_b.shape[0], stride);

    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1] * IMAGE_CHANNELS);

    CLBlastConvolution2D(&host_a, &host_b, &host_c, stride);

    SaveImg(output_file, &host_c);

    if (CheckImg(&answer, &host_c) != CL_SUCCESS)
    {
        fprintf(stderr, "Image check failed.\n");
        return -1;
    }

    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}
