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

    // Local memory tiles (optimize for memory coalescing)
    __local int Atile[TILE_SIZE][TILE_SIZE];
    __local int Btile[TILE_SIZE][TILE_SIZE];

    // Get thread indices
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    // Initialize accumulator
    int sum = 0;

    // Calculate number of tiles needed
    const int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;

    // Process each tile
    for (int tile = 0; tile < numTiles; tile++) {
        const int tileOffset = tile * TILE_SIZE;

        // Load tile from matrix A
        const int aRow = globalRow;
        const int aCol = tileOffset + localCol;
        
        // Vectorized load (int4 instead of int)
        if (aRow < numARows && aCol < numAColumns) {
            *((int4 *)&Atile[localRow][localCol]) = *((__global int4 *)&A[aRow * numAColumns + aCol]);
        } else {
            Atile[localRow][localCol] = 0;
        }

        // Load tile from matrix B
        const int bRow = tileOffset + localRow;
        const int bCol = globalCol;

        if (bRow < numBRows && bCol < numBColumns) {
            *((int4 *)&Btile[localRow][localCol]) = *((__global int4 *)&B[bRow * numBColumns + bCol]);
        } else {
            Btile[localRow][localCol] = 0;
        }

        // Ensure all threads have loaded their data
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Atile[localRow][k] * Btile[k][localCol];
        }

        // Ensure computation is complete before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the result only if within bounds
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = sum;
    }
}
