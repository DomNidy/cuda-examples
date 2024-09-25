#include <stdio.h>

#define N 512 // num of blocks

__global__ void add(int *a, int *b, int *c)
{
    printf("running on blockIdx=(%d,%d,%d), a loc=%p, b loc=%p, c loc=%p\n", blockIdx.x, blockIdx.y, blockIdx.z, a, b, c);
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
    // printf("(blockIdx: (%d,%d,%d) a points to: %p\n", blockIdx.x, blockIdx.y, blockIdx.z, a);
    // printf("(blockIdx: (%d,%d,%d) a points to: %p\n", blockIdx.x, blockIdx.y, blockIdx.z, a);
    // printf("Executed add() with args a=%d, b=%d, and yielded c=%d, (c mem addr: %p)\n", *a, *b, *c, c);
}

void printVec(const void *vec, size_t elementSize)
{
    printf("[ ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", *((int *)vec + i));
    }
    printf("]\n");
}

int main(void)
{
    int size = sizeof(int);

    // host copies of a,b,c
    // allocate memory to store vectors
    int *a = (int *)calloc(N, size);
    int *b = (int *)calloc(N, size);
    int *c = (int *)calloc(N, size);

    int *d_a, *d_b, *d_c; // device copies of a,b,c

    if (a == NULL || b == NULL || c == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    // assign some random values to our vectors
    for (int i = 0; i < N; i++)
    {
        *(a + i) = (5 + i) % 3;
        *(b + i) = (5 + i) % 7;
    }

    printf("Vectors initialized to:\n");
    printVec(a, size);
    printVec(b, size);
    printVec(c, size);

    // allocate space on the device for our vector copies
    // but remember, we cant dereference these from host code or pass to host code
    // we can only use these pointers in our kernels running on blocks on the gpu
    cudaMalloc((void **)&d_a, N * size);
    cudaMalloc((void **)&d_b, N * size);
    cudaMalloc((void **)&d_c, N * size);

    // copy inputs to the device
    // this moves data into the region of memory we allocated
    // on the gpu earlier (with cudaMalloc)
    cudaMemcpy(d_a, a, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * size, cudaMemcpyHostToDevice);

    // pasing the pointers to our region of memory containing
    // the data we copied earlier (with cudaMemcpy)
    add<<<N, 1>>>(d_a, d_b, d_c);

    // copy back to host, reading from the region of memory `d_c`
    // we allocated on the device, into the region on the host `c`
    cudaMemcpy(c, d_c, N * size, cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Result from add kernel:\n");
    printVec(c, size);

    return 0;
}