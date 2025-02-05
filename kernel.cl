__kernel void matrixMultiply(
    __global const short *restrict A,  // 16-bit input
    __global const short *restrict B,  // 16-bit input
    __global int *restrict C,          // 32-bit output
    const unsigned int numARows, 
    const unsigned int numAColumns,
    const unsigned int numBRows, 
    const unsigned int numBColumns,
    const unsigned int numCRows, 
    const unsigned int numCColumns) 
{
    #define TILE_SIZE 32
    #define SIMD_WIDTH 8  // AMD-specific optimization
    
    __local int Atile[TILE_SIZE][TILE_SIZE];
    __local int Btile[TILE_SIZE][TILE_SIZE];
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    
    int sum = 0;
    const int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;
    
    // AMD Wavefront optimization
    const int wavefrontIdx = localRow % SIMD_WIDTH;
    
    for (int tile = 0; tile < numTiles; ++tile) {
        const int tileOffset = tile * TILE_SIZE;
        
        // Coalesced load with 16-bit elements
        const int aCol = tileOffset + localCol;
        Atile[localRow][localCol] = (globalRow < numARows && aCol < numAColumns) ?
            (int)A[globalRow * numAColumns + aCol] : 0;
            
        const int bRow = tileOffset + localRow;
        Btile[localCol][localRow] = (bRow < numBRows && globalCol < numBColumns) ?
            (int)B[bRow * numBColumns + globalCol] : 0;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Warp shuffling optimization (AMD)
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k += SIMD_WIDTH) {
            int a_val = Atile[localRow][k + wavefrontIdx];
            int b_val = Btile[localCol][k + wavefrontIdx];
            
            // SIMD-wide reduction
            sum += a_val * b_val;
            sum += __shfl_down(sum, 1);
            sum += __shfl_down(sum, 2);
            sum += __shfl_down(sum, 4);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = sum;
    }
}
