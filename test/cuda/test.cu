#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cout << "Error: " << __FILE__ << ":" << __LINE__ << ", "; \
        std::cout << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

void test_pageable(size_t size, int iterations)
{
    std::cout << "\n=== Pageable Memory Test ===\n";

    float* h_mem = (float*)malloc(size);
    float* d_mem;

    CHECK(cudaMalloc(&d_mem, size));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        CHECK(cudaMemcpy(h_mem, d_mem, size, cudaMemcpyDeviceToHost));
    }

    CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(end - start).count();

    double gb = (double)size * iterations / 1e9;

    std::cout << "Time: " << sec << " s\n";
    std::cout << "Bandwidth: " << gb / sec << " GB/s\n";

    free(h_mem);
    cudaFree(d_mem);
}

void test_pinned(size_t size, int iterations)
{
    std::cout << "\n=== Pinned Memory Test ===\n";

    float* h_mem;
    float* d_mem;

    CHECK(cudaHostAlloc(&h_mem, size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&d_mem, size));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        CHECK(cudaMemcpyAsync(
            h_mem,
            d_mem,
            size,
            cudaMemcpyDeviceToHost
        ));
    }

    CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(end - start).count();
    double gb = (double)size * iterations / 1e9;

    std::cout << "Time: " << sec << " s\n";
    std::cout << "Bandwidth: " << gb / sec << " GB/s\n";

    cudaFreeHost(h_mem);
    cudaFree(d_mem);
}

void test_pipeline(size_t chunk_size, int iterations)
{
    std::cout << "\n=== Double Buffer Pipeline Test ===\n";

    float* d_buf[2];
    float* pinned_buf[2];

    CHECK(cudaMalloc(&d_buf[0], chunk_size));
    CHECK(cudaMalloc(&d_buf[1], chunk_size));

    CHECK(cudaHostAlloc(&pinned_buf[0], chunk_size, cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&pinned_buf[1], chunk_size, cudaHostAllocDefault));

    // large pageable memory (simulate huge checkpoint storage)
    float* pageable = (float*)malloc(chunk_size * iterations);

    cudaStream_t copy_stream;
    CHECK(cudaStreamCreate(&copy_stream));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        int cur = i % 2;
        int prev = (i + 1) % 2;

        // simulate GPU compute
        CHECK(cudaMemsetAsync(d_buf[cur], 1, chunk_size));

        // GPU → pinned
        CHECK(cudaMemcpyAsync(
            pinned_buf[cur],
            d_buf[cur],
            chunk_size,
            cudaMemcpyDeviceToHost,
            copy_stream));

        // CPU memcpy previous buffer → pageable storage
        if (i > 0)
        {
            std::memcpy(
                pageable + (i - 1) * chunk_size / sizeof(float),
                pinned_buf[prev],
                chunk_size);
        }

        CHECK(cudaStreamSynchronize(copy_stream));
    }

    // copy last chunk
    std::memcpy(
        pageable + (iterations - 1) * chunk_size / sizeof(float),
        pinned_buf[(iterations - 1) % 2],
        chunk_size);

    auto end = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(end - start).count();
    double total_gb = (double)chunk_size * iterations / 1e9;

    std::cout << "Total time: " << sec << " s\n";
    std::cout << "Effective bandwidth: " << total_gb / sec << " GB/s\n";

    cudaFree(d_buf[0]);
    cudaFree(d_buf[1]);

    cudaFreeHost(pinned_buf[0]);
    cudaFreeHost(pinned_buf[1]);

    free(pageable);

    cudaStreamDestroy(copy_stream);
}

void test_double_buffer(size_t size, int iterations)
{
    std::cout << "\n=== Double Buffer Async Test ===\n";

    float *h_buf1, *h_buf2;
    float *d_mem;

    CHECK(cudaHostAlloc(&h_buf1, size, cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&h_buf2, size, cudaHostAllocDefault));

    CHECK(cudaMalloc(&d_mem, size));

    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        if (i % 2 == 0)
        {
            CHECK(cudaMemcpyAsync(
                h_buf1,
                d_mem,
                size,
                cudaMemcpyDeviceToHost,
                stream1));
        }
        else
        {
            CHECK(cudaMemcpyAsync(
                h_buf2,
                d_mem,
                size,
                cudaMemcpyDeviceToHost,
                stream2));
        }
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    auto end = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(end - start).count();
    double gb = (double)size * iterations / 1e9;

    std::cout << "Time: " << sec << " s\n";
    std::cout << "Bandwidth: " << gb / sec << " GB/s\n";

    cudaFreeHost(h_buf1);
    cudaFreeHost(h_buf2);
    cudaFree(d_mem);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

int main()
{
    size_t size = 256 * 1024 * 1024; // 256MB
    int iterations = 100;

    std::cout << "Transfer size: " << size / 1024 / 1024 << " MB\n";
    std::cout << "Iterations: " << iterations << "\n";

    test_pageable(size, iterations);
    test_pinned(size, iterations);
    test_double_buffer(size, iterations);
    test_pipeline(size, iterations);

    return 0;
}