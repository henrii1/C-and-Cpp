#include <stdio.h>

// __global__ indicate function should run on gpu
__global__ void hello(){

    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);

}

int main(){
    hello<<<2,2>>>();   //2,2 represents the block, threads enabled for this computation
    
    cudaDeviceSynchronize();           //synchronize GPU computation with that of the CPU

}