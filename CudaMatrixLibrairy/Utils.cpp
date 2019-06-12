#include "Utils.h"

void Utils::validateCUDA() {
	int count;
	cudaGetDeviceCount(&count);
	
	if (count == 0) {
		std::cerr << "No CUDA devices present!\n";
		exit(1);
	}
}

std::string Utils::getCUDAName(int devideId) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devideId);
	return std::string (prop.name);
}

void Utils::print(float* matrix, unsigned int rows, unsigned int cols){

	for (unsigned int i = 0; i < rows; i++) {
		for (unsigned int j = 0; j < cols; j++) {
			std::cout << matrix[i * cols + j] << "\t";
		}
		std::cout << "\n";
	}


}



