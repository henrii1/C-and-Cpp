#include <stdio.h>

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

    const size_t DSIZE = 16384;
    const int block_size = 256;

    //matrix row-sum kernel
    __global__ void row_sums(const float *A, float *sums, size_t ds){
        
        int idx = blockIdx.x;    //block id is idx
        if (idx < ds){           //for each block in grid
            __shared__ float sdata[block_size];
            int tid = threadIdx.x;
            sdata[tid] = 0.0f;
            size_t tidx = tid;    //threadIdx
            __syncthreads();

            size_t tidx = tid + idx*BLOCK_SIZE;
            size_t end = min(tidx + BLOCK_SIZE, ds);   //fill them up considering global index

            while(tidx < end) {
                sdata[tid] += A[tidx];      //using the global index to initialize shared memory
                tidx += blockDim.x;
            }
            __syncthreads();
            //we've initialized the shared memory
            for (unsigned int s=blockDim.x/2; s>0; s>>=1){
                if (tid < s)   //sweep reduction
                    sdata[tid] += sdata[tid + s];
            }
            if (tid == 0) sums[idx] = sdata[0];

        }
    }
 //matrix column-sum kernel
        __global__ void col_sums(const float *A, float *sums, size_t ds){
            int idx = threadIdx.x+blockIdx.x*blockDim.x;
            if (idx < ds){
                float sum = 0.0f;
                for (size_t i = 0; i<ds; i++)
                    sum += A[idx+ds*i];
                sum[idx] = sum;
            }
        }

bool validate(float *data, size_t sz){
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;}
    return true;
}
int main(){

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE*DSIZE];  // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sums, DSIZE*sizeof(float));  // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  //cuda processing sequence step 1 is complete
  row_sums<<<DSIZE, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("row sums correct!\n");
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("column sums correct!\n");
  return 0;
}
  
