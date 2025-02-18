__kernel void convolution2D(
    __global int * inputData,
    __global int * outputData,
    __constant int * maskData,
    int width, int height, int maskWidth, int imageChannels, int stride) {
    
    // Get the index of the current element
    int j = get_global_id(0); // x-coordinate (width index)
    int i = get_global_id(1); // y-coordinate (height index)
    
    // Check if this thread is within the image dimensions
    if (i < height && j < width) {
        int maskRadius = maskWidth / 2;
        
        // Process each channel (R, G, B)
        for (int k = 0; k < imageChannels; k++) {
            int accum = 0;
            
            // Convolve the mask with the input data
            for (int y = -maskRadius; y <= maskRadius; y++) {
                for (int x = -maskRadius; x <= maskRadius; x++) {
                    int xOffset = j + x;
                    int yOffset = i + y;
                    
                    // Check bounds for VALID padding
                    if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {
                        // Get image pixel value
                        int imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                        
                        // Get mask value
                        int maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                        
                        // Accumulate the product
                        accum += imagePixel * maskValue;
                    }
                }
            }
            
            // Store the result in output
            outputData[(i * width + j) * imageChannels + k] = accum;
        }
    }
}
