#include <stdio.h>

//error macro
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


const size_t DSIZE = 16384;     //matrix side dim
const int block_size = 256;     //total dim (max is 1024)

//matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds){
    int idx = threadIdx.x + blockIdx.x*blockDim.x
    if (idx < ds){
        float sum = 0.0f;
        for (size_t i=0; i<ds; i++)
            sum += A[idx+ds*i];    //sums rows of matrix
        sums[idx] = sum     // shape = ds, 1
    }
}

//matrix col-sum kernel
__global__ void col_sums(const float *A, float *sums, size_t ds){
    int idx = threadIdx.x + blockIdx.x*blockDim.x
    if(idx < ds){
        float sum = 0.0f;
        for(size_t i=0; i<ds; i++)
            sum +=[idx+ds*i];
        sums[idx] = sum;
    }
}

bool validate(float *data, size_t sz){
    for(size_t i=0; i<sz; i++)
    if (data[i] != (float)sz) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;}
    return true;    
}

int main(){
    
    float *h_A, *h_sums, *d_A, *d_sums;
    h_A = new float[DSIZE*DSIZE];
    h_sums = new float[DSIZE*DSIZE];

    for (int i=0; i<DSIZE*DSIZE; i++)
        h_A[i] = 1.0f;

    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_sums, DSIZE*DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc Error");

    //copy
    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("H2D error");

    row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("Row sum Kernel Failure");

    //copy
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("D2H copy failure or Kernel launch failure");

    if(!validate(h_sums, DSIZE)) return -1;       //not an expected return type
    printf("row sums correct\n");    // if 0 is returned

    
    //clear memory
    cudaMemset(d_sums, 0, DSIZE*sizeof(float));

    col_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("column sum kernel failure");

    //COPY RESULT
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("D2H copy error or Kernel failure");

    if (!validate(h_sums, DSIZE)) return -1; 
  printf("column sums correct!\n");
  return 0;

}
