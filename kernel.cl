__kernel void matrixMultiply(
    __global const int *restrict A, 
    __global const int *restrict B, 
    __global int *restrict C,
    const unsigned int numARows, 
    const unsigned int numAColumns,
    const unsigned int numBRows, 
    const unsigned int numBColumns,
    const unsigned int numCRows, 
    const unsigned int numCColumns) 
{
    #define TILE_SIZE 16
    #define PADDING 1  // Anti-bank-conflict padding
    
    // Local memory tiles with bank conflict avoidance
    __local int Atile[TILE_SIZE][TILE_SIZE + PADDING];
    __local int Btile[TILE_SIZE][TILE_SIZE + PADDING];
    
    // Thread indices
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    
    // Accumulator with faster register allocation
    int sum = 0;
    
    // Tile processing
    const int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < numTiles; ++tile) {
        const int tileOffset = tile * TILE_SIZE;
        
        // Coalesced A tile load
        const int aCol = tileOffset + localCol;
        Atile[localRow][localCol] = (globalRow < numARows && aCol < numAColumns) 
            ? A[globalRow * numAColumns + aCol] 
            : 0;
        
        // Coalesced B tile load with transposed storage
        const int bRow = tileOffset + localRow;
        Btile[localCol][localRow] = (bRow < numBRows && globalCol < numBColumns) 
            ? B[bRow * numBColumns + globalCol] 
            : 0;
        
        // Memory fence for local data
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Unrolled computation with reduced address calculations
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Atile[localRow][k] * Btile[localCol][k];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Final store with coalesced write
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = sum;
    }
}
