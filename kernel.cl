__kernel void convolution2D(
    __global int * inputData,
    __global int * outputData,
    __constant int * maskData,
    int width,
    int height,
    int maskWidth,
    int imageChannels,
    int stride) {
    
    // Get the index of the current element
    int j = get_global_id(0); // column/width index
    int i = get_global_id(1); // row/height index
    int channels = 3; // As per instructions, we can assume channels is always 3
    
    // VALID padding means we only compute output for positions where the kernel fits entirely
    int maskRadius = maskWidth / 2;
    
    // Check if this thread is within the valid output dimensions (VALID padding)
    if (i < height && j < width) {
        // Perform convolution for each channel
        for (int k = 0; k < channels; k++) {
            int accum = 0;
            
            // Convolve the mask with the input data
            for (int y = -maskRadius; y <= maskRadius; y++) {
                for (int x = -maskRadius; x <= maskRadius; x++) {
                    int xOffset = j + x;
                    int yOffset = i + y;
                    
                    // Check bounds - this implements VALID padding
                    if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {
                        int imagePixel = inputData[(yOffset * width + xOffset) * channels + k];
                        int maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                        accum += imagePixel * maskValue;
                    }
                }
            }
            
            // Store the result in output
            outputData[(i * width + j) * channels + k] = accum;
        }
    }
}
