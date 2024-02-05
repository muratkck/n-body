#include "cuda_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */
void checkCuda(cudaError_t result) {
     if (result != cudaSuccess) {
        printf("CUDA Runtime Error: %s\n",cudaGetErrorString(result));
     }
}

__global__
void bodyForce(Body *p, float dt, int n) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride)
  {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

__global__ 
void calculatePosition(Body *p, float dt, int n) {
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = index; i < n; i += stride) {
        p[i].x += p[i].vx*dt;
        p[i].y += p[i].vy*dt;
        p[i].z += p[i].vz*dt;
    }
}



int main(const int argc, const char** argv) {

  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you generate ./nbody report files
  int nBodies = 2<<15;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);   // ASCII to Integer

  // The assessment will pass hidden initialized values to check for correctness.
  // You should not make changes to these files, or else the assessment will not work.
  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *host_buf;
  Body *device_buf;

  checkCuda(cudaMalloc(&device_buf, bytes));
  checkCuda(cudaMallocHost(&host_buf, bytes));
  Body *p = (Body*)host_buf;

  read_values_from_file(initialized_values, host_buf, bytes);
    
  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  double totalTime = 0.0;
    
  checkCuda(cudaMemcpyAsync(device_buf, p, bytes, cudaMemcpyHostToDevice));
    
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    bodyForce<<<numberOfBlocks, threadsPerBlock>>>(device_buf, dt, nBodies); // compute interbody forces
    checkCuda(cudaGetLastError());

    calculatePosition<<<numberOfBlocks, threadsPerBlock>>>(device_buf, dt, nBodies);
    checkCuda(cudaGetLastError());

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed; 
  }
    
  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

  checkCuda(cudaMemcpyAsync(p, device_buf, bytes, cudaMemcpyDeviceToHost));
  
  write_values_to_file(solution_values, host_buf, bytes);

  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  checkCuda(cudaFree(device_buf));
  checkCuda(cudaFreeHost(host_buf));
}
