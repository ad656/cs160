__kernel void convolution2D(
    __global int *inputData, __global int *outputData, __constant int *maskData,
    int width, int height, int maskWidth, int imageChannels) {
    
    // Get global work-item indices
    int j = get_global_id(0); // Column index (X coordinate)
    int i = get_global_id(1); // Row index (Y coordinate)
    
    // Calculate mask radius (integer division)
    int maskRadius = maskWidth / 2;
    
    // Ensure output is within valid bounds for VALID padding
    if (i >= maskRadius && i < height - maskRadius && j >= maskRadius && j < width - maskRadius) {
        for (int k = 0; k < imageChannels; k++) { // Iterate over R, G, B channels
            float accum = 0.0f;
            
            // Apply convolution mask
            for (int y = -maskRadius; y <= maskRadius; y++) {
                for (int x = -maskRadius; x <= maskRadius; x++) {
                    int xOffset = j + x;
                    int yOffset = i + y;
                    
                    // Compute index for input and mask
                    int inputIdx = (yOffset * width + xOffset) * imageChannels + k;
                    int maskIdx = (y + maskRadius) * maskWidth + (x + maskRadius);
                    
                    // Accumulate weighted sum
                    accum += inputData[inputIdx] * maskData[maskIdx];
                }
            }
            
            // Store result in output array (Clamp between 0 and 1 as per instructions)
            int outIdx = (i * width + j) * imageChannels + k;
            outputData[outIdx] = clamp(accum, 0.0f, 1.0f);
        }
    }
}
