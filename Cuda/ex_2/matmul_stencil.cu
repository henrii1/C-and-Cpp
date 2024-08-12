#include <stdio.h>
#include <time.h>

//error checking
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


const int DSIZE = 8192;    //matrix size==grid size
const int block_size = 32;  //thread per dim. total(32*32)
const float A_val = 3.0f;   //f enforces the RHS float
const float B_val = 2.0f;

//matrix mul kernel using shared
__global__ void mmul(const float *A, const float *B, const float *C, int ds){

    //shared cache memory
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y = blockIdx.y*blockDim.y;

    if ((idx < ds) && (idy < ds)){
        float temp = 0;
        for (int i=0; i<ds/block_size; i++){   //one i per block

            //load into shared memory
            As[threadIdx.y][threadIdx.x] = A[idy * ds + (i*block_size + threadId.x)];  //unrolled A. parrallel per block
            Bs[threadIdx.y][threadIdx.x] = B[(i*block_size + threadIdx.y)*ds + idx];


            //synchronize per block
            __syncthreads();

            //running sum
            for (int k=0; k<block_size; k++)
                temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];  //dot product
            __syncthreads();


        }

        // copy to result. note that each parrallel[i] recieved a copy of temp
        C[idy*ds+idx] = temp;
    }
    

}

int main(){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;

    t0 = clock();

    h_A = new float[DSIZE*DSIZE];
    h_B = new float[DSIZE*DSIZE];
    h_C = new float[DSIZE*DSIZE];

    for (int i=0; i<DSIZE*DSIZE; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    //Initialize time
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    //cuda memory
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));   //works like malloc
    cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));

    cudaCheckErrors("cuda Malloc failure");

    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B;, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaCheckErrors("copy from H2D");

    //kernel launch
    dim3 block(block_size, block_size);
    dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("Kernel Failure");

    //copy result to host
    cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    //time
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t2sum);


    //verify results
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}
    printf("Success!\n"); 
    return 0;
}

