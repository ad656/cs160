kernel void convolution2D(
    global const int *inputData,
    global int *outputData,
    constant int *maskData,
    int width,
    int height,
    int maskWidth,
    int imageChannels,
    int stride) {

    int maskRadius = maskWidth / 2;
    int outputHeight = height - maskWidth + 1;
    int outputWidth = width - maskWidth + 1;

    int output_j = get_global_id(0);
    int output_i = get_global_id(1);

    if (output_i >= outputHeight || output_j >= outputWidth) {
        return;
    }

    for (int k = 0; k < imageChannels; k++) {
        int accum = 0;

        for (int dy = 0; dy < maskWidth; dy++) {
            for (int dx = 0; dx < maskWidth; dx++) {
                int input_y = output_i + dy;
                int input_x = output_j + dx;

                int input_index = (input_y * width + input_x) * imageChannels + k;
                int pixel_value = inputData[input_index];

                int mask_index = dy * maskWidth + dx;
                int mask_value = maskData[mask_index];

                accum += pixel_value * mask_value;
            }
        }

        int output_index = (output_i * outputWidth + output_j) * imageChannels + k;
        outputData[output_index] = accum;
    }
}
