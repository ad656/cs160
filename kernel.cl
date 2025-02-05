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
        sum += Atile[localRow][0] * Btile[0][localCol];
        sum += Atile[localRow][1] * Btile[1][localCol];
        sum += Atile[localRow][2] * Btile[2][localCol];
        sum += Atile[localRow][3] * Btile[3][localCol];
        sum += Atile[localRow][4] * Btile[4][localCol];
        sum += Atile[localRow][5] * Btile[5][localCol];
        sum += Atile[localRow][6] * Btile[6][localCol];
        sum += Atile[localRow][7] * Btile[7][localCol];
        sum += Atile[localRow][8] * Btile[8][localCol];
        sum += Atile[localRow][9] * Btile[9][localCol];
        sum += Atile[localRow][10] * Btile[10][localCol];
        sum += Atile[localRow][11] * Btile[11][localCol];
        sum += Atile[localRow][12] * Btile[12][localCol];
        sum += Atile[localRow][13] * Btile[13][localCol];
        sum += Atile[localRow][14] * Btile[14][localCol];
        sum += Atile[localRow][15] * Btile[15][localCol];
        
        // Ensure computation is complete before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the result only if within bounds
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = sum;
    }
}
