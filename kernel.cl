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
    
    
    // Local memory tiles
    __local int Atile[16][16];
    __local int Btile[16][16];
    
    // Get thread indices
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    
    // Initialize accumulator
    int sum = 0;
    
    // Calculate number of tiles needed
    const int numTiles = (numAColumns + 15) / 16;
    
    // Process each tile
    for (int tile = 0; tile < numTiles; tile++) {
        const int tileOffset = tile * 16;
        
        // Load tile from matrix A
        const int aRow = globalRow;
        const int aCol = tileOffset + localCol;
        
        if (aRow < numARows && aCol < numAColumns) {
            Atile[localRow][localCol] = A[aRow * numAColumns + aCol];
        } else {
            Atile[localRow][localCol] = 0;
        }
        
        // Load tile from matrix B
        const int bRow = tileOffset + localRow;
        const int bCol = globalCol;
        
        if (bRow < numBRows && bCol < numBColumns) {
            Btile[localRow][localCol] = B[bRow * numBColumns + bCol];
        } else {
            Btile[localRow][localCol] = 0;
        }
        
        // Ensure all threads have loaded their data
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product for this tile  
        sum += Atile[localRow][k] * Btile[k][localCol];
        sum += Atile[localRow][k+1] * Btile[k+1][localCol];
        sum += Atile[localRow][k+2] * Btile[k+2][localCol];
        sum += Atile[localRow][k+3] * Btile[k+3][localCol];
        sum += Atile[localRow][k+4] * Btile[k+4][localCol];
        sum += Atile[localRow][k+5] * Btile[k+5][localCol];
        sum += Atile[localRow][k+6] * Btile[k+6][localCol];
        sum += Atile[localRow][k+7] * Btile[k+7][localCol];
        sum += Atile[localRow][k+8] * Btile[k+8][localCol];
        sum += Atile[localRow][k+9] * Btile[k+9][localCol];
        sum += Atile[localRow][k+10] * Btile[k+10][localCol];
        sum += Atile[localRow][k+11] * Btile[k+11][localCol];
        sum += Atile[localRow][k+12] * Btile[k+12][localCol];
        sum += Atile[localRow][k+13] * Btile[k+13][localCol];
        sum += Atile[localRow][k+14] * Btile[k+14][localCol];
        sum += Atile[localRow][k+15] * Btile[k+15][localCol];
        
        // Ensure computation is complete before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the result only if within bounds
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = sum;
    }
}
