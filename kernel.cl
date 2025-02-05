#define TILE_SIZE 8
__kernel void matrixMultiply(
		__global const int *A, __global const int *B, __global int *C,
		const unsigned int numARows, const unsigned int numAColumns,
		const unsigned int numBRows, const unsigned int numBColumns,
		const unsigned int numCRows, const unsigned int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	int i = get_local_id(0);
	int j = get_local_id(1);
	int row = get_global_id(0);
	int col = get_global_id(1);
	int lsize_i = get_local_size(0);
	int lsize_j = get_local_size(1);
	short isActive = 0;

	int sum = 0;
	__local int tile_A[TILE_SIZE*TILE_SIZE];
	__local int tile_B[TILE_SIZE*TILE_SIZE];

	/* 
	 * at E: cntr = 0, i = 2, j = 1, row = 2, col = 1, TILE_SIZE = 3
	 * row * numAColumns + col = 
	 * col * numBColumns + row = 
	 * 
	 e e e f f f g		e e e * *
	 e e e f f f g		e e Ë * *
	 e E e f f f g		e e e * *
	 * * * * * * *		f f f * *
	 * * * * * * *		f f f * *
	 f f f * *
	 g g g * *
	 */
	int cntr = 0;
	isActive = row < numARows && col < numBColumns;
	while(cntr * TILE_SIZE < numAColumns){
		// copy my part into the tile
		if(row < numARows && (j+cntr*TILE_SIZE < numAColumns))
			tile_A[i * TILE_SIZE + j] = A[row * numAColumns + (j+cntr*TILE_SIZE)];
		else 
			tile_A[i * TILE_SIZE + j] = 0;

		if(col < numBColumns && (i+cntr*TILE_SIZE < numBRows))
			tile_B[i * TILE_SIZE + j] = B[(i + cntr*TILE_SIZE) * numBColumns + col];
		else 
			tile_B[i * TILE_SIZE + j] = 0;

		// wait for others
		barrier(CLK_LOCAL_MEM_FENCE);

		/*if(isActive){*/
			// put partial sum in sum var
			for(int x = 0; x < TILE_SIZE; x++) { // assuming work groups are square
				int locA = tile_A[i * TILE_SIZE + x];
				int locB = tile_B[x * TILE_SIZE + j];
				sum += locA * locB;
			}
		/*}*/
		// on to the next one
		cntr++;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(row < numCRows && col < numCColumns)
		C[row * numCColumns + col] = sum;
	
}
