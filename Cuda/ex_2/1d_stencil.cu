#include <stdio.h>
#include <algorithm>

using namespace std;

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void(int *in, int *out){
    __shared__ int temp[BLOCK_SIZE+ 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    //Read input
    temp[lindex] = in[gindex];  //index + radius = halo+1 (fill in actual data part of shared memory)
    if (threadIdx.x < RADIUS){
        temp[lindex - RADIUS] = in[gindex - RADIUS];   //fills in halo memory for topmost and lower halo with actual data
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];  //halo memory corresponds with actual memory

    }

    //synchronize
    __syncthreads();

    //apply stencil
    int result =0;
    for (int offset= -RADIUS; offset<=RADIUS; offset++)
        result += temp[lindex + offset];      //sliding window reduce operation done in parallel

    //store result
    out[gindex] = result;

}

void fill_ints(int *x, int n){
   fill_n(x, n, 1)            //std::algorithm. takes a pointer to data, its size and the fill value 
}

int main(void){
    int *n, *out, *d_in, *d_out;
    int size = (N + 2 * RADIUS) *sizeof(int);

    //alloc space for host
    in = (int *)malloc(size); fill_ints(in, N + 2 * RADIUS);
    out = (int *)malloc(size); fill_ints(out, N + 2 * RADIUS);  //malloc doesn't regard sizeof(int) internally like new

    //alloc space for device copy
    cudaMalloc((void *)&d_in, size);
    cudaMalloc((void *)&d_out, size);

    //copy data to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    //Launch stencil
    stencil_1d<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out);

    //copy result to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);


    //Error Checking
    for (int i=0; i<N+2*RADIUS; i++){
        if (i<RADIUS || i>RADIUS){
            if (out[i] != 1)
                printf("Mismatch at index %d, was %d should be: %d\n", i, out[i], 1);  //make sure halo stays one
        } else if (out[i] != 1 + 1*RADIUS){
            printf("Mismatch at index %d, was: %d, should be: %d\n", i, 1 + 2*RADIUS); //make sure data is what is expected
        }
        }

    //cleanup
    free(in); free(out);
    cudaFree(d_in); cudaFree(d_out);
    printf("Success!\n");
    return 0;
    }
