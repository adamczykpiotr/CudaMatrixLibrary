#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <ctime>
#include <thread>
#include <vector>
#include "Utils.h"

namespace Host{

	namespace Workers {
		__host__ void addWorker(float* source0, float* source1, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols);
		__host__ void substractWorker(float* source0, float* source1, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols);
		__host__ void multiplyWorker(float* source0, float* source1, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols0, unsigned int cols1);
		__host__ void multiplyWorkerScalar(float* source0, float scalar, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols);
		__host__ void detWorker(float* source, unsigned int size, float* det);
		__host__ void transposeWorker(float* source, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int rows, unsigned int cols);
	};
	
	__host__ clock_t add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols);
	__host__ clock_t add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads);


	__host__ clock_t substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols);
	__host__ clock_t substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads);


	__host__ clock_t multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1);
	__host__ clock_t multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1, unsigned int nThreads);

	__host__ clock_t multiply(float* source0, float scalar, float* destination, unsigned int rows, unsigned int cols);
	__host__ clock_t multiply(float* source0, float scalar, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads);

	// <determinant helper functions>
	__host__ int getSign(unsigned int i);
	__host__ void createSubMatrix(float* source, float* destination, unsigned int colRemove, unsigned int rowRemove, unsigned int size);
	// </determinant helper functions>

	__host__ clock_t determinant(float* source, float * det, unsigned int size);
	__host__ clock_t determinant(float* source, float * det, unsigned int size, unsigned int nThreads);

	__host__ clock_t transpose(float * source, float* destination, unsigned int rows, unsigned int cols);
	__host__ clock_t transpose(float* source, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads);
	
};

