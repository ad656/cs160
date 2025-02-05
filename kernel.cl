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
    #define VEC_SIZE 4  // Process 4 elements at once
    
    __local int Atile[TILE_SIZE][TILE_SIZE + 1];  // Padded
    __local int Btile[TILE_SIZE][TILE_SIZE + 1];  // Padded
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    
    int sum = 0;
    const int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;

    // Vectorized version for tile loading
    for (int tile = 0; tile < numTiles; ++tile) {
        const int tileOffset = tile * TILE_SIZE;
        
        // Vector load for A
        if (globalRow < numARows) {
            const int aCol = tileOffset + localCol * VEC_SIZE;
            #pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i) {
                Atile[localRow][localCol * VEC_SIZE + i] = 
                    (aCol + i < numAColumns) ? 
                    A[globalRow * numAColumns + aCol + i] : 0;
            }
        }
        
        // Vector load for B
        if (globalCol < numBColumns) {
            const int bRow = tileOffset + localRow * VEC_SIZE;
            #pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i) {
                Btile[localCol][localRow * VEC_SIZE + i] = 
                    (bRow + i < numBRows) ? 
                    B[(bRow + i) * numBColumns + globalCol] : 0;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        // Vectorized computation
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += VEC_SIZE) {
            sum += Atile[localRow][k] * Btile[localCol][k];
            sum += Atile[localRow][k+1] * Btile[localCol][k+1];
            sum += Atile[localRow][k+2] * Btile[localCol][k+2];
            sum += Atile[localRow][k+3] * Btile[localCol][k+3];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = sum;
    }
}
