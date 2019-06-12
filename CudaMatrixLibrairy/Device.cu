#include "Device.h"

unsigned int blockSize = 16;
unsigned int blockSizeSqrt = 4;

__global__ void Device::Workers::multiplyWorker(float* source0, float* source1, float* destination, int rows0, int cols0, int cols1){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.;
	if (col < cols1 && row < rows0) {

		for (int i = 0; i < cols0; i++) {
			sum += source0[row * cols0 + i] * source1[i * cols1 + col];
		}

		destination[row * cols1 + col] = sum;
	}
}

__global__ void Device::Workers::multiplyWorkerScalar(float* source0, float scalar, float* destination, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < cols && row < rows) {

		destination[row * cols + col] = source0[row * cols + col] * scalar;
	}
}

__global__ void Device::Workers::transposeWorker(float* source0, float* destination, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < cols && row < rows) {

		destination[col * rows + row] = source0[row * cols + col];
	}
}

__global__ void Device::Workers::addWorker(float* source0, float* source1, float* destination, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < cols && row < rows) {

		destination[row * cols + col] = source0[row * cols + col] + source1[row * cols + col];
	}
}

__global__ void Device::Workers::substractWorker(float* source0, float* source1, float* destination, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < cols && row < rows) {

		destination[row * cols + col] = source0[row * cols + col] - source1[row * cols + col];
	}
}

clock_t Device::multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1){

	clock_t start = clock();

	//Memory allocation on the deive
	float* deviceSource0, * deviceSource1, * deviceDestination;
	cudaMalloc((void**)& deviceSource0, sizeof(float) * rows0 * cols0);
	cudaMalloc((void**)& deviceSource1, sizeof(float) * cols0 * cols1);
	cudaMalloc((void**)& deviceDestination, sizeof(float) * rows0 * cols1);

	//Copy matrix
	cudaMemcpy(deviceSource0, source0, sizeof(float) * rows0 * cols0, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSource1, source1, sizeof(float) * cols0 * cols1, cudaMemcpyHostToDevice);

	unsigned int gridCols = blockSizeSqrt;
	unsigned int grideviceDestinationols;
	if (rows0 / gridCols > cols1 / gridCols) {
		grideviceDestinationols = rows0 / gridCols;

		if (rows0 % gridCols != 0) {
			grideviceDestinationols++;
		}

	} else {
	
		grideviceDestinationols = cols1 / gridCols;

		if (cols1 % gridCols != 0) {
			grideviceDestinationols++;
		}
	}


	dim3 dimGrid(grideviceDestinationols, grideviceDestinationols, 1);
	dim3 dimBlock(gridCols, gridCols,1);

	Device::Workers::multiplyWorker << <dimGrid, dimBlock >> > (deviceSource0, deviceSource1, deviceDestination, rows0, cols0, cols1);

	// Transfer results
	cudaMemcpy(destination, deviceDestination, sizeof(float) * rows0 * cols1, cudaMemcpyDeviceToHost);
	destination = deviceDestination;

	cudaDeviceSynchronize();

	return clock() - start;
}

clock_t Device::multiply(float* source0, float scalar, float* destination, int rows, int cols) {

	clock_t start = clock();

	//Memory allocation on the deive
	float* deviceSource0, * deviceDestination;
	cudaMalloc((void**)& deviceSource0, sizeof(float) * rows * cols);
	cudaMalloc((void**)& deviceDestination, sizeof(float) * rows * cols);

	//Copy matrix
	cudaMemcpy(deviceSource0, source0, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

	unsigned int gridCols = blockSizeSqrt;
	unsigned int grideviceDestinationols = rows / gridCols;

	if (rows % gridCols != 0) {
		grideviceDestinationols++;
	}

	dim3 dimGrid(grideviceDestinationols, grideviceDestinationols, 1);
	dim3 dimBlock(gridCols, gridCols, 1);

	Device::Workers::multiplyWorkerScalar << <dimGrid, dimBlock >> > (deviceSource0, scalar, deviceDestination, rows, cols);


	// Transfer results
	cudaMemcpy(destination, deviceDestination, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	destination = deviceDestination;

	cudaDeviceSynchronize();

	return clock() - start;
}

clock_t Device::transpose(float* source0, float* destination, unsigned int rows, unsigned int cols) {
	clock_t start = clock();

	//Memory allocation on the deive
	float* deviceSource0, * deviceDestination;
	cudaMalloc((void**)& deviceSource0, sizeof(float) * rows * cols);
	cudaMalloc((void**)& deviceDestination, sizeof(float) * rows * cols);

	//Copy matrix
	cudaMemcpy(deviceSource0, source0, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

	unsigned int gridCols = blockSizeSqrt;
	unsigned int grideviceDestinationols = rows / gridCols;

	if (rows % gridCols != 0) {
		grideviceDestinationols++;
	}

	dim3 dimGrid(grideviceDestinationols, grideviceDestinationols, 1);
	dim3 dimBlock(gridCols, gridCols, 1);

	Device::Workers::transposeWorker<< <dimGrid, dimBlock >> > (deviceSource0, deviceDestination, rows, cols);

	// Transfer results
	cudaMemcpy(destination, deviceDestination, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	destination = deviceDestination;

	cudaDeviceSynchronize();

	return clock() - start;
}

clock_t Device::add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols) {

	clock_t start = clock();

	//Memory allocation on the deive
	float* deviceSource0, * deviceSource1, * deviceDestination;
	cudaMalloc((void**)& deviceSource0, sizeof(float) * rows * cols);
	cudaMalloc((void**)& deviceSource1, sizeof(float) * rows * rows);
	cudaMalloc((void**)& deviceDestination, sizeof(float) * rows * cols);

	//Copy matrix
	cudaMemcpy(deviceSource0, source0, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSource1, source1, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

	unsigned int gridCols = blockSizeSqrt;
	unsigned int grideviceDestinationols = rows / gridCols;

	if (rows % gridCols != 0) {
		grideviceDestinationols++;
	}

	dim3 dimGrid(grideviceDestinationols, grideviceDestinationols, 1);
	dim3 dimBlock(gridCols, gridCols, 1);

	Device::Workers::addWorker << <dimGrid, dimBlock >> > (deviceSource0, deviceSource1, deviceDestination, rows, cols);


	// Transfer results
	cudaMemcpy(destination, deviceDestination, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	destination = deviceDestination;

	cudaDeviceSynchronize();

	return clock() - start;
}


clock_t Device::substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols) {

	clock_t start = clock();

	//Memory allocation on the deive
	float* deviceSource0, * deviceSource1, * deviceDestination;
	cudaMalloc((void**)& deviceSource0, sizeof(float) * rows * cols);
	cudaMalloc((void**)& deviceSource1, sizeof(float) * rows * rows);
	cudaMalloc((void**)& deviceDestination, sizeof(float) * rows * cols);

	//Copy matrix
	cudaMemcpy(deviceSource0, source0, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSource1, source1, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

	unsigned int gridCols = blockSizeSqrt;
	unsigned int grideviceDestinationols = rows / gridCols;

	if (rows % gridCols != 0) {
		grideviceDestinationols++;
	}

	dim3 dimGrid(grideviceDestinationols, grideviceDestinationols, 1);
	dim3 dimBlock(gridCols, gridCols, 1);

	Device::Workers::addWorker << <dimGrid, dimBlock >> > (deviceSource0, deviceSource1, deviceDestination, rows, cols);


	// Transfer results
	cudaMemcpy(destination, deviceDestination, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	destination = deviceDestination;

	cudaDeviceSynchronize();

	return clock() - start;
}
