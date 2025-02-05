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
    #define TILE_SIZE 32  // Increased tile size
    #define PADDING 1
    
    // Double buffered local memory
    __local int Atile[2][TILE_SIZE][TILE_SIZE + PADDING];
    __local int Btile[2][TILE_SIZE][TILE_SIZE + PADDING];
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    
    int sum = 0;
    const int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;
    
    // Preload first tile
    int loadTile = 0;
    {
        const int tileOffset = 0;
        const int aCol = tileOffset + localCol;
        Atile[loadTile][localRow][localCol] = 
            (globalRow < numARows && aCol < numAColumns) ? 
            A[globalRow * numAColumns + aCol] : 0;
            
        const int bRow = tileOffset + localRow;
        Btile[loadTile][localCol][localRow] = 
            (bRow < numBRows && globalCol < numBColumns) ? 
            B[bRow * numBColumns + globalCol] : 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int tile = 0; tile < numTiles; ++tile) {
        const int nextTile = (tile + 1) % 2;
        const int currentTile = tile % 2;
        
        // Async load next tile while computing current
        if(tile < numTiles - 1) {
            const int tileOffset = (tile + 1) * TILE_SIZE;
            const int aCol = tileOffset + localCol;
            Atile[nextTile][localRow][localCol] = 
                (globalRow < numARows && aCol < numAColumns) ? 
                A[globalRow * numAColumns + aCol] : 0;
                
            const int bRow = tileOffset + localRow;
            Btile[nextTile][localCol][localRow] = 
                (bRow < numBRows && globalCol < numBColumns) ? 
                B[bRow * numBColumns + globalCol] : 0;
        }
        
        // Compute current tile
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; ++k) {
            sum += Atile[currentTile][localRow][k] * 
                   Btile[currentTile][localCol][k];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (globalRow < numCRows && globalCol < numCColumns) {
        C[globalRow * numCColumns + globalCol] = sum;
    }
}
