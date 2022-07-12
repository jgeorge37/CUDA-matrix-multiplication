#include <stdio.h>
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 18432;
int block_size = 32;
const float A_val = 3.0f;
const float B_val = 2.0f;

// matrix multiply kernel - dynamic shared memory
__global__ void mmul(const float *A, const float *B, float *C, int ds)
{
  extern __shared__ float s[];
  float *A_data = s;
  float *B_data = &A_data[blockDim.x*blockDim.y];
  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  int x, y;
  float temp = 0;
  for (int i = 0; i < gridDim.x; i++) {
    x = i * blockDim.x + threadIdx.x;
    y = i * blockDim.x + threadIdx.y;
    A_data[threadIdx.y*blockDim.x+threadIdx.x] = x < ds && idy < ds ? A[idy*ds+x] : 0;
    B_data[threadIdx.y*blockDim.x+threadIdx.x] = y < ds && idx < ds ? B[y*ds+idx] : 0;
    __syncthreads();
    for (int j = 0; j < blockDim.x; j++)
      temp += A_data[threadIdx.y * blockDim.x + j] * B_data[j * blockDim.x + threadIdx.x];
    __syncthreads();
  }
  if ((idx < ds) && (idy < ds)) C[idy*ds+idx] = temp;
}

int main(int argc, char *argv[])
{

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // for timing
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  if (argc == 2) {
      block_size = atoi(argv[1]);
      if (block_size <= 0) {
          fprintf(stderr, "Error: block_size should be >= 1\n");
          exit (1);
      }
  }

  // start timing
  t0 = clock();

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");

  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Launch kernel
  dim3 block(block_size, block_size);
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block, block.x*block.y*2*sizeof(float)>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy D2H failure");

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute took %f seconds\n", t2sum);

  // Cleanup
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

  // Verify results
  for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}
  printf("Success!\n");
  return 0;
}
