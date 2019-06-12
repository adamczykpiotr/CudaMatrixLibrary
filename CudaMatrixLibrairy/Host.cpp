#include "Host.h"

__host__ clock_t Host::add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols) {

	clock_t start = clock();

	for (unsigned int i = 0u; i < rows; i++) {
		for (unsigned int j = 0u; j < cols; j++) {
			destination[i * cols + j] = source0[i * cols + j] + source1[i * cols + j];
		}
	}

	return clock() - start;

}

__host__ void Host::Workers::addWorker(float* source0, float* source1, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols) {
	for (unsigned int i = rowStart; i < rowEnd; i++) {
		for (unsigned int j = 0u; j < cols; j++) {
			destination[i * cols + j] = source0[i * cols + j] + source1[i * cols + j];
		}
	}
}

__host__ void Host::Workers::substractWorker(float* source0, float* source1, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols) {
	for (unsigned int i = rowStart; i < rowEnd; i++) {
		for (unsigned int j = 0; j < cols; j++) {
			destination[i * cols + j] = source0[i * cols + j] - source1[i * cols + j];
		}
	}
}

__host__ void Host::Workers::multiplyWorker(float* source0, float* source1, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols0, unsigned int cols1) {
	for (unsigned int i = rowStart; i < rowEnd; i++) {
		for (unsigned int j = 0u; j < cols1; j++) {

			float sum = 0.f;

			for (unsigned int k = 0; k < cols0; k++) {
				sum += source0[i * cols0 + k] * source1[k * cols1 + j];
			}

			destination[i * cols1 + j] = sum;
		}
	}
}

__host__ void Host::Workers::multiplyWorkerScalar(float* source0, float scalar, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int cols) {

	for (unsigned int i = rowStart; i < rowEnd; i++) {
		for (unsigned int j = 0u; j < cols; j++) {
			destination[i * cols + j] = source0[i * cols + j] * scalar;
		}
	}
}




__host__ void Host::Workers::detWorker(float* source, unsigned int size, float* det) {

	if (size < 4) {
		switch (size) {
		case 1:
			(*det) = source[0];

		case 2:
			(*det) = source[0] * source[3] - source[1] * source[2];

		case 3:
			(*det) = source[0] * source[size + 1] * source[2 * size + 2] + source[1] * source[size + 2] * source[2 * size + 0] + source[2] * source[size + 0] * source[2 * size + 1];
			(*det) -= (source[2] * source[size + 1] * source[2 * size + 0] + source[0] * source[size + 2] * source[2 * size + 1] + source[1] * source[size + 0] * source[2 * size + 2]);
		}

		return;

	}

	unsigned int newSize = size - 1;

	std::vector<float*> matrices;
	float* dets = new float[size];

	for (unsigned int i = 0; i < size; i++) {
		float* t = new float[newSize*newSize];
		matrices.push_back(t);
		createSubMatrix(source, matrices[i], i, 0, size);
	}

	for (unsigned int i = 0; i < size; i++) {
		detWorker(matrices[i], newSize, &dets[i]);
	}


	(*det) = 0.f;
	for (int i = 0; i < size; i++) {
		(*det) += source[i] * getSign(i) * dets[i];
	}

}

__host__ void Host::Workers::transposeWorker(float* source, float* destination, unsigned int rowStart, unsigned int rowEnd, unsigned int rows, unsigned int cols) {


		for (unsigned int i = rowStart; i < rowEnd; i++) {
			for (int j = 0; j < cols; j++) {
			destination[i + j * rows] = source[(i * cols) + j];
		}
	}

}

__host__ clock_t Host::add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads) {

	clock_t start = clock();

	std::vector<std::thread> threads(nThreads);

	unsigned int avgIntervalLength = static_cast<unsigned int>(ceil(rows / nThreads));

	unsigned int rowIterator = 0u;
	for (unsigned int i = 0u; i < nThreads-1; i++) {
		threads[i] = std::thread(Host::Workers::addWorker, source0, source1, destination, rowIterator, rowIterator+avgIntervalLength, cols);
		rowIterator += avgIntervalLength;
	}
	threads[nThreads-1] = std::thread(Host::Workers::addWorker, source0, source1, destination, avgIntervalLength*(nThreads-1), rows, cols);


	for (unsigned int i = 0u; i < nThreads ; i++) {
		threads[i].join();
	}
	   
	return clock() - start;

}

__host__  clock_t Host::substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols) {

	clock_t start = clock();

	for (unsigned int i = 0u; i < rows; i++) {
		for (unsigned int j = 0u; j < cols; j++) {
			destination[i * cols + j] = source0[i * cols + j] - source1[i * cols + j];
		}
	}

	return clock() - start;
}

__host__ clock_t Host::substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads) {

	clock_t start = clock();

	std::vector<std::thread> threads(nThreads);

	unsigned int avgIntervalLength = static_cast<unsigned int>(ceil(rows / nThreads));

	unsigned int rowIterator = 0u;
	for (unsigned int i = 0u; i < nThreads - 1; i++) {
		threads[i] = std::thread(Host::Workers::addWorker, source0, source1, destination, rowIterator, rowIterator + avgIntervalLength, cols);
		rowIterator += avgIntervalLength;
	}
	threads[nThreads - 1] = std::thread(Host::Workers::addWorker, source0, source1, destination, avgIntervalLength * (nThreads - 1), rows, cols);


	for (unsigned int i = 0u; i < nThreads; i++) {
		threads[i].join();
	}

	return clock() - start;
}


__host__ clock_t Host::multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1) {
	
	clock_t start = clock();

	for (unsigned int i = 0u; i < rows0; i++) {
		for (unsigned int j = 0u; j < cols1; j++) {

			float sum = 0.f;

			for (unsigned int k = 0; k < cols0; k++) {
				sum += source0[i * cols0 + k] * source1[k * cols1 + j];
			}

			destination[i * cols1 + j] = sum;
		}
	}

	return clock() - start;

}

__host__ clock_t Host::multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1, unsigned int nThreads) 
{
	clock_t start = clock();

	std::vector<std::thread> threads(nThreads);

	unsigned int avgIntervalLength = static_cast<unsigned int>(ceil(rows0 / nThreads));

	unsigned int rowIterator = 0u;
	for (unsigned int i = 0; i < nThreads - 1; i++) {
		threads[i] = std::thread(Host::Workers::multiplyWorker, source0, source1, destination, rowIterator, rowIterator + avgIntervalLength, cols0, cols1);
		rowIterator += avgIntervalLength;
	}
	threads[nThreads - 1] = std::thread(Host::Workers::multiplyWorker, source0, source1, destination, rowIterator, rows0, cols0, cols1);


	for (unsigned int i = 0u; i < nThreads; i++) {
		threads[i].join();
	}

	return clock() - start;
}

__host__ clock_t Host::multiply(float* source0, float scalar, float* destination, unsigned int rows, unsigned int cols) {

	clock_t start = clock();

	for (unsigned int i = 0u; i < rows; i++) {
		for (unsigned int j = 0u; j < cols; j++) {
			destination[i * cols + j] = source0[i * cols + j] * scalar;
		}
	}

	return clock() - start;

}

__host__ clock_t Host::multiply(float* source0, float scalar, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads) {

	clock_t start = clock();

	std::vector<std::thread> threads(nThreads);

	unsigned int avgIntervalLength = static_cast<unsigned int>(ceil(rows / nThreads));

	unsigned int rowIterator = 0u;
	for (unsigned int i = 0u; i < nThreads - 1; i++) {
		threads[i] = std::thread(Host::Workers::multiplyWorkerScalar, source0, scalar, destination, rowIterator, rowIterator + avgIntervalLength, cols);
		rowIterator += avgIntervalLength;
	}
	threads[nThreads - 1] = std::thread(Host::Workers::multiplyWorkerScalar, source0, scalar, destination, avgIntervalLength * (nThreads - 1), rows, cols);


	for (unsigned int i = 0u; i < nThreads; i++) {
		threads[i].join();
	}

	return clock() - start;
}


__host__ int Host::getSign(unsigned int i) {

	if ((i + 1 + 1) % 2 == 0) {
		return 1;
	}

	return -1;

}

__host__ void Host::createSubMatrix(float* source, float* destination, unsigned int colRemove, unsigned int rowRemove, unsigned int size) {
	unsigned int i2 = 0u, j2 = 0u;
	for (unsigned int i = 0u; i < size; i++) {

		if (i == rowRemove) {
			continue;
		}

		j2 = 0u;
		for (unsigned int j = 0u; j < size; j++) {

			if (j == colRemove) {
				continue;
			}

			destination[i2 * (size - 1) + j2] = source[i * size + j];

			j2++;

		}

		i2++;

	}

}

__host__ clock_t Host::determinant(float* source, float * det, unsigned int size) {

	clock_t start = clock();

	if (size < 4) {
		switch (size) {
		case 1:
			(*det) = source[0];

		case 2:
			(*det) = source[0] * source[3] - source[1] * source[2];

		case 3:
			(*det) = source[0] * source[size + 1] * source[2 * size + 2] + source[1] * source[size + 2] * source[2 * size + 0] + source[2] * source[size + 0] * source[2 * size + 1];
			(*det) -= (source[2] * source[size + 1] * source[2 * size + 0] + source[0] * source[size + 2] * source[2 * size + 1] + source[1] * source[size + 0] * source[2 * size + 2]);
		}
		return clock() - start;
	}

	unsigned int newSize = size - 1;

	std::vector<float*> matrices;
	size_t vecSize = size;
	vecSize *= size;

	std::thread* pool = new std::thread[size];
	float* dets = new float[size];

	for (unsigned int i = 0; i < size; i++) {
		float* t = new float[vecSize];
		matrices.push_back(t);
		createSubMatrix( source, matrices[i], i, 0, size);
	}

	for (unsigned int i = 0; i < size; i++) {
		Workers::detWorker(matrices[i], newSize, &dets[i]);
	}

	(*det) = 0;
	for (int i = 0; i < size; i++) {
		(*det) += source[i] * getSign(i) * dets[i];
	}


	return clock() - start;
}

__host__ clock_t Host::determinant(float* source, float* det, unsigned int size, unsigned int nThreads) {
	//input threads quantity doesn't matter

	clock_t start = clock();

	if (size < 4) {
		switch (size) {
		case 1:
			(*det) = source[0];

		case 2:
			(*det) = source[0] * source[3] - source[1] * source[2];

		case 3:
			(*det) = source[0] * source[size + 1] * source[2 * size + 2] + source[1] * source[size + 2] * source[2 * size + 0] + source[2] * source[size + 0] * source[2 * size + 1];
			(*det) -= (source[2] * source[size + 1] * source[2 * size + 0] + source[0] * source[size + 2] * source[2 * size + 1] + source[1] * source[size + 0] * source[2 * size + 2]);
		}
		return clock() - start;
	}
	
	unsigned int newSize = size - 1;

	std::vector<float*> matrices;
	size_t vecSize = size;
	vecSize *= size;

	std::thread* pool = new std::thread[size];
	float* dets = new float[size];

	for (unsigned int i = 0; i < size; i++) {
		float* t = new float[vecSize];
		matrices.push_back(t);
		pool[i] = std::thread(createSubMatrix, source, matrices[i], i, 0, size);
	}

	for (unsigned int i = 0; i < size; i++) {
		pool[i].join();
	}

	for (unsigned int i = 0; i < size; i++) {
		pool[i] = std::thread(Workers::detWorker, matrices[i], newSize, &dets[i]);
	}

	for (unsigned int i = 0; i < size; i++) {
		pool[i].join();
	}

	(*det) = 0;
	for (int i = 0; i < size; i++) {
		(*det) += source[i] * getSign(i) * dets[i];
	}

	return clock() - start;
}

__host__ clock_t Host::transpose(float* source, float* destination, unsigned int rows, unsigned int cols) {

	clock_t start = clock();

	for (int j = 0; j < cols; j++) {
		for (unsigned int i = 0; i < rows; i++) {
			destination[i + j * rows] = source[(i * cols) + j];
		}
	}		

	return clock() - start;

}

__host__ clock_t Host::transpose(float* source, float* destination, unsigned int rows, unsigned int cols, unsigned int nThreads) {
	clock_t start = clock();

	std::vector<std::thread> threads(nThreads);

	unsigned int avgIntervalLength = static_cast<unsigned int>(ceil(rows / nThreads));

	unsigned int rowIterator = 0u;
	for (unsigned int i = 0u; i < nThreads - 1; i++) {
		threads[i] = std::thread(Host::Workers::transposeWorker,source, destination, rowIterator, rowIterator + avgIntervalLength, rows, cols);
		rowIterator += avgIntervalLength;
	}
	threads[nThreads - 1] = std::thread(Host::Workers::transposeWorker, source, destination, avgIntervalLength * (nThreads - 1), rows, rows, cols);


	for (unsigned int i = 0u; i < nThreads; i++) {
		threads[i].join();
	}

	return clock() - start;
}
