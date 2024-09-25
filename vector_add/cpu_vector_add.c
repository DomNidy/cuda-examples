#include <stdio.h>
#define N 1000 * 1000
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

    for (int i = 0; i < N; i++)
    {
        *(c + i) = a[i] + b[i];
    }

    int sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += c[i];
    }
    printf("noncuda=%d\n", sum);
    return 0;
}
