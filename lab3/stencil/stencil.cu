/**
 * stencil.cu: a simple 1d stencil on GPU and on CPU
 * 
 * Implement the basic stencil and make sure it works correctly.
 * Then, play with the code
 * - Experiment with block sizes, various RADIUSes and NUM_ELEMENTS.
 * - Measure the memory transfer time, estimate the effective memory bandwidth.
 * - Estimate FLOPS (floating point operations per second)
 * - Switch from float to double: how the performance changes?
*/

scp -r kk429629@students:/home/students/mismap/k/kk429629/HPC/lab3 ~/courses/HPC/lab3


#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <cmath>

#define RADIUS        3
#define NUM_ELEMENTS  1000 
#define BLOCK_WIDTH   10

#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

static void handleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

__global__ void stencil_1d(float *in, float *out) {
  //PUT YOUR CODE HERE
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < NUM_ELEMENTS) {
    out[i] = 0;
    int start = std::max(0, i - RADIUS);
    int end   = std::min(NUM_ELEMENTS - 1, i + RADIUS);
    for (int j = start; j <= end; j++) {
      out[i] += in[j];
    }
  }
}

void cpu_stencil_1d(float *in, float *out) {
  //PUT YOUR CODE HERE
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    int start = std::max(0, i - RADIUS);
    int end   = std::min(NUM_ELEMENTS - 1, i + RADIUS);
    for (int j = start; j <= end; j++) {
      out[i] += in[j];
    }
  }
}

bool check(float *out, float *dev_out) {
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    if (std::abs(out[i] - dev_out[i]) > std::numeric_limits<float>::epsilon()) {
      printf("Mismatch on index %d (expected = %f, actual = %f)\n", i, out[i], dev_out[i]);
      return false;
    }
  }
  return true;
}

int main() {
  //PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS
  float in[NUM_ELEMENTS], out[NUM_ELEMENTS], dev_out_cpy[NUM_ELEMENTS];
  float *dev_in, *dev_out;

  for (int i = 0; i < NUM_ELEMENTS; i++) {
    in[i] = 1;
  }
  
  cudaEvent_t gpu_start, gpu_stop;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  cudaEventRecord(gpu_start, 0);

  //PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
  cudaMalloc((void**)&dev_in,  NUM_ELEMENTS * sizeof(float));
  cudaMalloc((void**)&dev_out, NUM_ELEMENTS * sizeof(float));

  cudaMemcpy(dev_in,  in,  NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

  //PUT YOUR CODE HERE - KERNEL EXECUTION
  int num_blocks = (NUM_ELEMENTS + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  stencil_1d<<<num_blocks, NUM_ELEMENTS>>>(dev_in, dev_out);

  cudaCheck(cudaPeekAtLastError());

  //PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST
  struct timespec memtransfer_start, memtransfer_stop;  
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &memtransfer_start);
  cudaMemcpy(dev_out_cpy, dev_out, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &memtransfer_stop);
  double memoryTransferElapsedTime = (memtransfer_stop.tv_sec - memtransfer_start.tv_sec) * 1e3 + (memtransfer_stop.tv_nsec - memtransfer_start.tv_nsec) / 1e6;
  printf("Memory transfer time:  %.4f ms\n", memoryTransferElapsedTime);
  printf("Bandwidth:  %.4f Mb/s\n", 1e3 * NUM_ELEMENTS * sizeof(float) / memoryTransferElapsedTime);
  printf("FLOPS: %.4f\n", 1e3 * NUM_ELEMENTS * (2 * RADIUS + 1) / memoryTransferElapsedTime);

  cudaEventRecord(gpu_stop, 0);
  cudaEventSynchronize(gpu_stop);
  float gpuElapsedTime;
  cudaEventElapsedTime(&gpuElapsedTime, gpu_start, gpu_stop);
  printf("Total GPU execution time:  %.4f ms\n", gpuElapsedTime);
  printf("Memory transfer time:  %.4f ms\n", gpuElapsedTime);
  cudaEventDestroy(gpu_start);
  cudaEventDestroy(gpu_stop);

  //PUT YOUR CODE HERE - FREE DEVICE MEMORY  
  cudaFree(dev_in);
  cudaFree(dev_out);

  struct timespec cpu_start, cpu_stop;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
 
  cpu_stencil_1d(in, out);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
  double cpuElapsedTime = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf("CPU execution time:  %.4f ms\n", cpuElapsedTime);

  

  if (!check(out, dev_out_cpy)) {
    return 1;
  } else {
    printf("OK\n");
    return 0;
  }
}
