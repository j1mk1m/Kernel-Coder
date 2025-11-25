// Transpose shared_B to mitigate bank conflicts
__shared__ T shared_A[TILE_DIM][TILE_DIM];
__shared__ T shared_B[TILE_DIM][TILE_DIM];

// When loading B into shared memory
shared_B[threadIdx.x][threadIdx.y] = ...; // Transposed indices

// In the computation loop
sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];