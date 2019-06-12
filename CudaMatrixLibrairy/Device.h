#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>

namespace Device {

	namespace Workers {
		__global__ void addWorker(float* source0, float* source1, float* destination, int rows, int cols);

		__global__ void substractWorker(float* source0, float* source1, float* destination, int rows, int cols);

		__global__ void multiplyWorker(float* source0, float* source1, float* destination, int rows0, int cols0, int cols1);
		__global__ void multiplyWorkerScalar(float* source0, float scalar, float* destination, int rows, int cols);

		__global__ void transposeWorker(float* source0, float* destination, int rows, int cols);
	}
	
	clock_t add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols);
	clock_t substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols);

	clock_t multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1);
	clock_t multiply(float* source0, float scalar, float* destination, int rows, int cols);

	clock_t transpose(float* source0,float* destination, unsigned int rows, unsigned int cols);
	
};
