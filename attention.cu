#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void linear_kernel(float* input, float* weight, float* bias, float* output,
                              int batch, int seq, int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * seq * output_dim;

    if (idx < total_elements) {
        int b = idx / (seq * output_dim);
        int s = (idx % (seq * output_dim)) / output_dim;
        int o = idx % output_dim;
        float val = (bias != nullptr) ? bias[o] : 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            val += input[(b * seq + s) * input_dim + i] * weight[o * input_dim + i];
        }
        output[idx] = val;
    }
}

void qkv_compute_only(
    float* X_device,
    float* W_Q, float* W_K, float* W_V,
    float* Q_out, float* K_out, float* V_out,
    int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream
) {
    int threads = 256;
    int total_compute_elems = batch_size * seq_len * hidden_dim;
    int blocks = (total_compute_elems + threads - 1) / threads;

    // Compute Q for entire sequence
    linear_kernel<<<blocks, threads, 0, stream>>>(
        X_device, W_Q, nullptr, Q_out,
        batch_size, seq_len, hidden_dim, hidden_dim
    );

    // Compute K for entire sequence
    linear_kernel<<<blocks, threads, 0, stream>>>(
        X_device, W_K, nullptr, K_out,
        batch_size, seq_len, hidden_dim, hidden_dim
    );

    // Compute V for entire sequence
    linear_kernel<<<blocks, threads, 0, stream>>>(
        X_device, W_V, nullptr, V_out,
        batch_size, seq_len, hidden_dim, hidden_dim
    );

    cudaStreamSynchronize(stream);
}


void qkv_known_partition_overlap(
    float* X_device, float* K_host, float* V_host,
    float* W_Q, float* W_K, float* W_V,
    float* Q_out, float* K_out, float* V_out,
    int batch_size, int seq_len, int hidden_dim,
    int heads, int d_k, int partition,
    cudaStream_t stream_copy, cudaStream_t stream_compute
) {
    int computed_seq_len = seq_len - partition;
    size_t loaded_size_per_tensor = batch_size * partition * heads * d_k * sizeof(float);

    // Async memcpy in COPY stream
    cudaMemcpyAsync(K_out, K_host, loaded_size_per_tensor, cudaMemcpyHostToDevice, stream_copy);
    cudaMemcpyAsync(V_out, V_host, loaded_size_per_tensor, cudaMemcpyHostToDevice, stream_copy);

    int threads = 256;

    // Compute Q for entire sequence in COMPUTE stream
    int total_compute_elems_full = batch_size * seq_len * hidden_dim;
    int blocks_full = (total_compute_elems_full + threads - 1) / threads;
    linear_kernel<<<blocks_full, threads, 0, stream_compute>>>(
        X_device, W_Q, nullptr, Q_out,
        batch_size, seq_len, hidden_dim, hidden_dim
    );

    // Compute K,V for the smaller partition in COMPUTE stream
    int total_compute_elems = batch_size * computed_seq_len * hidden_dim;
    int blocks = (total_compute_elems + threads - 1) / threads;

    linear_kernel<<<blocks, threads, 0, stream_compute>>>(
        X_device + partition * hidden_dim, W_K, nullptr, K_out + partition * hidden_dim,
        batch_size, computed_seq_len, hidden_dim, hidden_dim
    );
    linear_kernel<<<blocks, threads, 0, stream_compute>>>(
        X_device + partition * hidden_dim, W_V, nullptr, V_out + partition * hidden_dim,
        batch_size, computed_seq_len, hidden_dim, hidden_dim
    );

    cudaStreamSynchronize(stream_copy);
    cudaStreamSynchronize(stream_compute);
}


void qkv_load(
    float* host_K, float* host_V,
    float* device_K, float* device_V,
    float* X_device, float* W_Q, float* Q_out,
    int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream
) {
    size_t tensor_bytes = batch_size * seq_len * hidden_dim * sizeof(float);

    cudaMemcpyAsync(device_K, host_K, tensor_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_V, host_V, tensor_bytes, cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (batch_size * seq_len * hidden_dim + threads - 1) / threads;

    linear_kernel<<<blocks, threads, 0, stream>>>(
        X_device, W_Q, nullptr, Q_out,
        batch_size, seq_len, hidden_dim, hidden_dim
    );

    cudaStreamSynchronize(stream);
}

int main(int argc, char* argv[]) {
    const int d_model = 512;
    const int seq_len = atoi(argv[1]);
    const int h = 8;
    const int d_k = d_model / h;
    const int batch_size = 32;
    const int partition = atoi(argv[2]);
    size_t tensor_size = batch_size * seq_len * d_model;
    size_t tensor_bytes = tensor_size * sizeof(float);

    // Host allocations (pinned memory)
    float *host_K, *host_V;
    cudaMallocHost(&host_K, tensor_bytes);
    cudaMallocHost(&host_V, tensor_bytes);
    for (size_t i = 0; i < tensor_size; i++) {
        host_K[i] = static_cast<float>(rand()) / RAND_MAX;
        host_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }


    // Device allocations
    float *dev_X, *dev_Q, *dev_K, *dev_V;
    cudaMalloc(&dev_X, tensor_bytes); cudaMemset(dev_X, 0, tensor_bytes);
    cudaMalloc(&dev_Q, tensor_bytes);
    cudaMalloc(&dev_K, tensor_bytes);
    cudaMalloc(&dev_V, tensor_bytes);

    // Weights allocation
    float *W_Q, *W_K, *W_V;
    cudaMalloc(&W_Q, d_model * d_model * sizeof(float));
    cudaMalloc(&W_K, d_model * d_model * sizeof(float));
    cudaMalloc(&W_V, d_model * d_model * sizeof(float));

    float *host_W = (float*)malloc(d_model * d_model * sizeof(float));
    for (int i = 0; i < d_model * d_model; i++)
        host_W[i] = static_cast<float>(rand()) / RAND_MAX;

    cudaMemcpy(W_Q, host_W, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_K, host_W, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_V, host_W, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);

    free(host_W);

    cudaStream_t stream_copy, stream_compute;
    cudaStreamCreate(&stream_copy);
    cudaStreamCreate(&stream_compute);

    // Use a default single stream for non-overlapping benchmarks clearly
    cudaStream_t default_stream;
    cudaStreamCreate(&default_stream);

    // Benchmark qkv_load (memory transfer dominant - use copy stream)
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, stream_copy);

    qkv_load(
        host_K, host_V, dev_K, dev_V, dev_X, W_Q, dev_Q,
        batch_size, seq_len, d_model, stream_copy
    );

    cudaEventRecord(stop, stream_copy);
    cudaStreamSynchronize(stream_copy);
    float ms_qkv_load;
    cudaEventElapsedTime(&ms_qkv_load, start, stop);
    printf("qkv_load took: %.3f ms\n", ms_qkv_load);

    // Benchmark qkv_compute_only (computation dominant - use compute stream)
    cudaEventRecord(start, stream_compute);

    qkv_compute_only(
        dev_X, W_Q, W_K, W_V,
        dev_Q, dev_K, dev_V,
        batch_size, seq_len, d_model, stream_compute
    );

    cudaEventRecord(stop, stream_compute);
    cudaStreamSynchronize(stream_compute);
    float ms_qkv_compute_only;
    cudaEventElapsedTime(&ms_qkv_compute_only, start, stop);
    printf("qkv_compute_only took: %.3f ms\n", ms_qkv_compute_only);

    // Benchmark qkv_known_partition_overlap (clearly uses both streams)
    cudaEventRecord(start, 0); // Record on default stream for accurate timing

    qkv_known_partition_overlap(
        dev_X, host_K, host_V, W_Q, W_K, W_V,
        dev_Q, dev_K, dev_V,
        batch_size, seq_len, d_model, h, d_k, partition,
        stream_copy, stream_compute
    );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait on default stream clearly
    float ms_qkv_partition_overlap;
    cudaEventElapsedTime(&ms_qkv_partition_overlap, start, stop);
    printf("qkv_known_partition_overlap took: %.3f ms\n", ms_qkv_partition_overlap);

    // Cleanup
    cudaFree(dev_X);
    cudaFree(dev_Q);
    cudaFree(dev_K);
    cudaFree(dev_V);
    cudaFree(W_Q);
    cudaFree(W_K);
    cudaFree(W_V);
    cudaFreeHost(host_K);
    cudaFreeHost(host_V);
    cudaStreamDestroy(stream_copy);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(default_stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
