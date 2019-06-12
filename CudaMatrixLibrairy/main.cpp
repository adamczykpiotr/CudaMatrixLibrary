#include <iostream>

#include "Utils.h"
#include "Launcher.h"

//http://www.bluebit.gr/matrix-calculator/calculate.aspx

int main() {

	Utils::validateCUDA();

	unsigned int rows0, cols0, rows1, cols1;

	
	rows0 = 2048u;
	cols0 = 1024u;

	rows1 = 1024u;
	cols1 = 512u;

	// allocate memory in host RAM
	float* hostMatrix0;
	float* hostMatrix1;
	float* hostMatrix2;
	float* hostMatrixSum;
	float* hostMatrixMul;

	cudaMallocHost((void**)& hostMatrix0, sizeof(float) * rows0 * cols0);
	cudaMallocHost((void**)& hostMatrix1, sizeof(float) * rows0 * cols0);
	cudaMallocHost((void**)& hostMatrix2, sizeof(float) * rows1 * cols1);

	cudaMallocHost((void**)& hostMatrixSum, sizeof(float) * rows0 * cols0);
	cudaMallocHost((void**)& hostMatrixMul, sizeof(float) * rows0 * cols1);

	srand(static_cast<unsigned int>(time(nullptr)));

	// initialize matrices
	for (unsigned int i = 0; i < rows0; ++i) {
		for (unsigned int j = 0; j < cols0; ++j) {
			hostMatrix0[i * cols0 + j] = static_cast<float>(i+j);
			hostMatrix1[i * cols0 + j] = static_cast<float>(j);
		}
	}

	for (unsigned int i = 0; i < rows1; ++i) {
		for (unsigned int j = 0; j < cols1; ++j) {
			hostMatrix2[i * cols1 + j] = static_cast<float>(j);
		}
	}

	Launcher::setSamples(1);
	Launcher::determinant(hostMatrix0, 9);
	Launcher::add(hostMatrix0, hostMatrix1, hostMatrixSum, rows0, cols0);
	Launcher::substract(hostMatrix0, hostMatrix1, hostMatrixSum, rows0, cols0);
	Launcher::multiply(hostMatrix0, hostMatrix1, hostMatrixMul, rows0, cols0, cols1);
	Launcher::multiply(hostMatrix0, 0.1f, hostMatrix1, rows0, cols0);
	Launcher::transpose(hostMatrix0, hostMatrix1, rows0, cols0);
	

	//Utils::print(hostMatrix0, rows0, cols0);

	return 0;
}