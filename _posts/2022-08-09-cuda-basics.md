---
layout: post
title: CUDA Basics
date: 2022-08-09 21:40:16
description: Notes on CUDA
tags: CUDA
categories: programming
---

Recently, during my internship at EPFL, I worked on a CUDA kernel for matrix orthogonalization using QR decomposition. I was new to CUDA programming. Thus, I decided to write these quick notes for my future self to help refresh fundamental concepts.

If you need to use your CUDA code with PyTorch, follow [this guide](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension) after reading this.

## How CUDA code is executed

The GPU can easily run multiple threads in parallel. Each thread needs to execute the same instruction. Thus, each thread reads the same intruction from a CUDA function (called kernel) and executes it. \\
What is the purpose of executing the same instruction multiple times in parallel? Well, the low-level instruction is the same, but each thread has some special registers with different values. With this difference as the starting point, each thread can read different pieces of memory, has different local results, etc. 

We will see an example soon, but first, let me define some important concepts:
 - **block**: a group of threads. When we launch a CUDA kernel, we specify the number of blocks and how many threads each block has. Threads on the same block can communicate with a shared memory and synchronization primitives since they are executed on the same Streaming Multiprocessor (SM). A GPU has multiple SM thus, if the above features are not needed, we should consider packing the total number of threads in multiple blocks. 
 - **warp**: a group of 32 threads from the same block that are executed simultaneously. Not all threads that we create are executed simultaneously since we can create more threads than GPU cores. Threads are drawn out in groups of 32 (warp) to be executed together using the same clock. Thus, to maximize occupancy, we should define a number of threads per block that is a multiple of 32 (in practice a power of two, because the number of cores is always a power of two). Threads in the same warp share registers, allowing for some [warp-level primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Software-Perspective_for_thread_block.jpg/800px-Software-Perspective_for_thread_block.jpg)

Let's do an easy example to show these concepts. The following kernel computes the sum of two vectors a, b, and saves the result in vector c.

{% highlight CUDA linenos %}
__global__ 
void cuda_add(const float *a, const float *b, float *c, const int vecSize) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < vecSize){
    c[i] = a[i] + b[i]
  }
}
{% endhighlight %}

where *blockDim*, *blockIdx*, and *threadIdx* indicates the number of thread per block, the index of the block, and the index of the thread respectively. Those dimensions can be 3-dimensional; in this simple case, we can just work with the x-axis, with a 1-D number of blocks and threads per block. \\
The keyword `__global__` indicates that this function is a CUDA kernel and it can be launched from the CPU. Notice that the `__global__` function must be void, thus we should work with C++ pointers for the output. This is the common scenario: the CPU allocates and copies the input into the GPU memory, launches the kernel, wait for the completion, and reads the result from the output memory. Thus, we can launch the kernel with a function similar to the following one:
{% highlight C++ linenos %} float* add(const float *a, const float *b, const int vecSize) {
    float *d_a, *d_b, *d_c; // device (GPU) vectors

    size_t memSize = vecSize * sizeof(float);
    
    // allocate memory in GPU
    cudaMalloc((void **)&d_a, memSize);
    cudaMalloc((void **)&d_b, memSize);
    cudaMalloc((void **)&d_c, memSize);

    // copy inputs from CPU to GPU
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    // launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = ceil( (float)vecSize / (float)threadsPerBlock ); // include math.h
    cuda_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, vecSize);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);

    // allocate CPU memory for result and copy data from GPU
    float *c = (float *)malloc(memSize);
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    cudaFree(d_c);

    return c;
}
{% endhighlight %}

In the above example, we launch the kernel with 256 threads per block and a number of block that depends on the *vecSize*, so we have at least one thread per element. The best number of thread per block depends on the GPU and the application: it should be tuned empirically. The maximum number in recent architecure is 1024, thus it usually suffice to try few power of two to find the best value. In this example, each element of the resulting vector *c* has a thread to compute the result. For some input dimension, it is faster to have a thread handling multiple sum with a for loop. In this case, for best performances, each warp should access to a contiguous memory in order to be able to exploit the caching mechanism; we will give an example in the later section.
CUDA operations are non-blocking, but they are scheduled in a *stream*. It means that line 25 might be executed before the `cuda_add` kernel completion. The GPU executes the operation in the stream sequentially, so we are sure that `cudaFree` lines are executed only after the completion of the kernel. Instead, `cudaMemcpy` is a blocking operation (with `cudaMemcpyAsync` the non-blocking counterpart), thus the result we return will be consistent.
We can schedule CUDA operations on different streams to increase parallelism when there aren't ordering contraint. We can block the CPU, waiting for operation on the GPU using the function `cudaDeviceSynchronize` or `cudaStreamSynchronize`.


## Reduction pattern

We showed a simple pattern, where every element of the resulting vector can be computed independently from the others. 
Another common pattern is *reduction*, when partial results should be combined. The simplest example is the dot product, where the kernel should sum the product of elements. 
Writing an efficient kernel for reduction is not trivial (as starting point, you can have a look at [these slides](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)), but fortunately, we can rely on libraries such as *[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)* or *[CUB](https://nvlabs.github.io/cub/)*. The first one contains high performant kernel for common linear algebra operations.
For example, to compute the dot product, we can write something similar to this:

{% highlight C++ linenos %}#include <cublas_v2.h>

int main(){
    // create the cuBLAS library context
    cublasHandle_t h;
    cublasCreate(&h);

    ... // do something, create the cuda vectors d_a, d_b

    float result = 0;
    cublasDdot(h, vecSize, d_a, 1, d_b, 1, &result);

    ... // do soething else, use the result stored in the CPU variable result

    // destroy cuBLAS context
    cublasDestroy(h);
    return 0;
}
{% endhighlight %}

If you want to write a custom kernel that includes reductions along with other operations, you can rely on the CUB library, which contains useful block-wide and warp-wide primitive functions. It means, that instead of calling the function from the CPU, we can call it inside a kernel, where every thread calls it. For example, we can write a dot function:

{% highlight C++ linenos %}#include <cub/cub.cuh>

__global__ 
float dot(float *a, float *b, int vecSize){
    int tx = threadIdx.x;
    int unroll = ceil( (float)vecSize / (float)blockDim.x );
    int idx = (tx & -32u) * unroll + (tx & 31);

    float localProd = 0;
    for (int i = 0; i < unroll; ++i){
        if (idx < length){
          localProd += a[idx] * b[idx];
        }
        idx += 32;
    }

    int nWarp = ceil(blockDim.x / 32.0);
    __shared__ float tmp_storage[nWarp];
    float reduce = cub::BlockReduceSum(localProd, tmp_storage);

    // reduce contains the correct result only on thread 0
     __shared__ float dot;
    if (tx == 0) 
        dot = reduce;
    __syncthreads();

    return dot;
}
{% endhighlight %}

Remembering that `&` is the bitwise and operation, `tx & 31` mask every bit except the last 5, thus it computes the modulo 32 operation of `tx`. Instead, `tx & -32u` takes the other bits, thus it is the warp index multiplied by 32. With this in mind, line 7 computes the index for the current thread so warps can be unrolled on a contiguous memory, as shown in the following figure, where we have a vector of 128 elements and 64 threads, thus 2 warps and `unroll = 2`. In the first iteration of the loop in line 10, each thread of the first warp compute the `localProd` of the elements from 0 to 31. In the second iteration, they slide by 32 position and accumulate the product of elements. 

{% include figure.html path="assets/img/warp-unroll.jpg" class="img-fluid rounded" %}

After the for loop, each thread has a local variable `localProd` with the sum of the product of `unroll` elements; the sum of all `localProd` will give us the `dot` product of the two input vectors. To achieve this, we can rely on the `BlockReduceSum` provided by CUB that should be called by each thread with the local value and shared memory to store partial results. The shared memory is allocated using the keyword `__shared__` and it is a memory accessible by all threads of the block. As you may have noticed, to support reduction we allocated a shared array that is long as the number of warps. Thread on the same warp can reduce their local value with faster registers (see [these slides](https://on-demand.gputechconf.com/gtc/2013/presentations/S3174-Kepler-Shuffle-Tips-Tricks.pdf)), while different warps need to communicate through shared memory. The value returned by `BlockReduceSum` will be the correct result for thread 0 but will be a partial result for other threads. If we need the correct result in all threads, we can rely again on a shared value and let the first thread store the value there. Since different warps can be executed independently, we need to synchronize threads after the assignment; we can do it with the primitive `__syncthreads()`. In this way, every thread will wait until all the others in the same block reach this line. 
You may wonder how it is possible to have an `if` statement when all threads in a warp execute the same instruction simultaneously. This is handled by the compiler: other threads will execute `NOP` while thread 0 executes the branch. This is called *branch divergence* and, to improve performances, it should be avoided when possible. 

## Useful resources
There is much more to say about CUDA. A complete starting point is the [CUDA programming guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf).
Other specific topics can be found on the [documentation page](https://docs.nvidia.com/cuda/index.html).
University courses exist and provide material for a softer starting point, for example, the slides from [this Oxford course](https://people.maths.ox.ac.uk/gilesm/cuda/), that I report here for convenience:
 - [An introduction to CUDA](https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_01.pdf)
 - [Different memory and variable types](https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_02.pdf)
 - [Control flow and synchronisation](https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_03.pdf)
 - [Warp shuffles, and reduction / scan operations](https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_04.pdf)
 - [Libraries and tools](https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_05.pdf)
 - [Multiple GPUs, and odds and ends](https://people.maths.ox.ac.uk/gilesm/cuda/2019/lecture_06.pdf)
 - [Tackling a new CUDA application](https://people.maths.ox.ac.uk/gilesm/cuda/2019/lec7.pdf)
 - [Profiling and tuning applications](https://people.maths.ox.ac.uk/gilesm/cuda/2019/CUDA_Course_profiling2.pdf)