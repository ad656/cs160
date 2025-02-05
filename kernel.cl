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
    #define VECTOR_SIZE 4
    #define PADDING 1

    // Local memory tiles with padding to avoid bank conflicts
    __local int Atile[TILE_SIZE][TILE_SIZE + PADDING];
    __local int Btile[TILE_SIZE][TILE_SIZE + PADDING];

    // Thread indices
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    // Pre-calculate global offsets
    const int aRowOffset = globalRow * numAColumns;
    const int bColOffset = globalCol;

    // Use vector accumulator for better arithmetic intensity
    int4 sum_vec = (int4)(0, 0, 0, 0);
    
    // Process tiles
    const int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; ++tile) {
        const int tileOffset = tile * TILE_SIZE;
        
        // Vectorized load for matrix A
        const int aCol = tileOffset + localCol;
        if (globalRow < numARows && aCol < numAColumns) {
            Atile[localRow][localCol] = A[aRowOffset + aCol];
        } else {
            Atile[localRow][localCol] = 0;
        }
        
        // Vectorized load for matrix B with transpose
        const int bRow = tileOffset + localRow;
        if (bRow < numBRows && globalCol < numBColumns) {
            Btile[localCol][localRow] = B[bRow * numBColumns + bColOffset];
        } else {
            Btile[localCol][localRow] = 0;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Vectorized computation with loop unrolling
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k += VECTOR_SIZE) {
            if (k + VECTOR_SIZE <= TILE_SIZE) {
                int4 a_vec = (int4)(
                    Atile[localRow][k],
                    Atile[localRow][k+1],
                    Atile[localRow][k+2],
                    Atile[localRow][k+3]
                );
                
                int4 b_vec = (int4)(
                    Btile[localCol][k],
                    Btile[localCol][k+1],
                    Btile[localCol][k+2],
                    Btile[localCol][k+3]
                );
                
                sum_vec += a_vec * b_vec;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Reduce vector sum
    int final_sum = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
    
    // Coalesced write to global memory
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = final_sum;
    }
}
