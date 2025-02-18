__kernel void convolution2D(
    __global int * inputData, __global int * outputData, __constant int * maskData,
    int width, int height, int maskWidth, int imageChannels, int stride) {
    
    // Get the global ID of the work-item (x, y coordinates)
    int j = get_global_id(0);
    int i = get_global_id(1);
    
    // Calculate mask radius
    int maskRadius = maskWidth / 2;
    
    // Check boundaries
    if (i < height && j < width) {
        // Calculate output coordinates based on stride
        int outRow = i / stride;
        int outCol = j / stride;
        
        // Skip if not on stride
        if (i % stride != 0 || j % stride != 0) {
            return;
        }
        
        // For each channel
        for (int k = 0; k < imageChannels; k++) {
            int accum = 0;
            
            // Convolve with mask
            for (int y = -maskRadius; y <= maskRadius; y++) {
                for (int x = -maskRadius; x <= maskRadius; x++) {
                    // Calculate input position
                    int xOffset = j + x;
                    int yOffset = i + y;
                    
                    // Boundary check for input
                    if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {
                        // Get input pixel at offset position for this channel
                        int inputIdx = (yOffset * width + xOffset) * imageChannels + k;
                        int imagePixel = inputData[inputIdx];
                        
                        // Get corresponding mask value
                        int maskIdx = (y + maskRadius) * maskWidth + (x + maskRadius);
                        int maskValue = maskData[maskIdx];
                        
                        // Accumulate
                        accum += imagePixel * maskValue;
                    }
                }
            }
            
            // Write output
            int outIdx = (outRow * (width / stride) + outCol) * imageChannels + k;
            outputData[outIdx] = clamp(accum, 0, 255); // Assuming 8-bit integers
        }
    }
}
