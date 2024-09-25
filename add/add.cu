#include <stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
    printf("(blockIdx: (%d,%d,%d) a points to: %p\n", blockIdx.x, blockIdx.y, blockIdx.z, a);
    printf("Executed add() with args a=%d, b=%d, and yielded c=%d, (c mem addr: %p)\n", *a, *b, *c, c);
}

int main(void)
{
    int a, b, c;          // host copies of a,b,c
    int *d_a, *d_b, *d_c; // device copies of a,b,c
    int size = sizeof(int);

    a = 2;
    b = 7;

    // allocate space for the device copies
    // after making these calls, our pointers d_a, d_b, d_c
    // will point to newly allocated memory on the gpu
    // but remember, we cant dereference these from host code or pass to host code
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy inputs to the device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // launch the add() kernel on GPU
    add<<<3, 1>>>(d_a, d_b, d_c);

    // copy back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Result from add kernel: %d", c);

    return 0;
}