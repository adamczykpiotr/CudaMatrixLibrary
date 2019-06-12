#include "Launcher.h"

size_t Launcher::nThreads = static_cast<size_t>(std::thread::hardware_concurrency());
unsigned int Launcher::samples = 3;

void Launcher::setNThreads(size_t nThreads_){
	nThreads = nThreads_;
}

void Launcher::setSamples(unsigned int samples_) {
	samples = samples_;
}

void Launcher::add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols) {

	std::vector<std::pair<float, std::string>> results;
	size_t size = nThreads + 2ull;
	results.reserve(size);

	clock_t time;
	float avgTime;
	
	// GPU
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Device::add(source0, source1, destination, rows, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "GPU " + Utils::getCUDAName()));

	// Single threaded
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Host::add(source0, source1, destination, rows, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(1) + "T (SINGLE)"));

	//Multi threaded
	for (unsigned int threadId = 1u; threadId < nThreads+1; threadId++) {

		time = 0;
		for (unsigned int i = 0u; i < samples; i++) {
			time += Host::add(source0, source1, destination, rows, cols, threadId);
		}
		avgTime = static_cast<float>(time);
		avgTime /= samples;
		results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(threadId) + "T (MULTI) "));
	}

	//Print out results
	std::cout << "Addition results:\n";
	for (size_t i = 0ull; i < size; i++) {
		std::cout << results[i].second << ":\t" << results[i].first << "ms\n";\
	}
	std::cout << "\n";

}

void Launcher::substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols) {

	std::vector<std::pair<float, std::string>> results;
	size_t size = nThreads + 2ull;
	results.reserve(size);

	clock_t time;
	float avgTime;

	// GPU
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Device::substract(source0, source1, destination, rows, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "GPU " + Utils::getCUDAName()));

	// Single threaded
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Host::substract(source0, source1, destination, rows, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(1) + "T (SINGLE)"));


	//Multi threaded
	for (unsigned int threadId = 1u; threadId < nThreads + 1; threadId++) {

		time = 0;
		for (unsigned int i = 0u; i < samples; i++) {
			time += Host::substract(source0, source1, destination, rows, cols, threadId);
		}
		avgTime = static_cast<float>(time);
		avgTime /= samples;
		results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(threadId) + "T (MULTI) "));
	}

	//Print out results
	std::cout << "Substraction results:\n";
	for (size_t i = 0ull; i < size; i++) {
		std::cout << results[i].second << ":\t" << results[i].first << "ms\n";
	}
	std::cout << "\n";

}

void Launcher::multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1) {

	std::vector<std::pair<float, std::string>> results;
	size_t size = nThreads + 2ull;
	results.reserve(size);

	clock_t time;
	float avgTime;

	// GPU
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Device::multiply(source0, source1, destination, rows0, cols0, cols1);
	}

	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "GPU " + Utils::getCUDAName()));

	// Single threaded
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Host::multiply(source0, source1, destination, rows0, cols0, cols1);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(1) + "T (SINGLE)"));

	//Multi threaded
	for (unsigned int threadId = 1u; threadId < nThreads + 1; threadId++) {

		time = 0;
		for (unsigned int i = 0u; i < samples; i++) {
			time += Host::multiply(source0, source1, destination, rows0, cols0, cols1, threadId);
		}
		avgTime = static_cast<float>(time);
		avgTime /= samples;
		results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(threadId) + "T (MULTI) "));
	}

	//Print out results
	std::cout << "Multiplication (by another matrix) results:\n";
	for (size_t i = 0ull; i < size; i++) {
		std::cout << results[i].second << ":\t" << results[i].first << "ms\n";
	}
	std::cout << "\n";

}

void Launcher::multiply(float* source0, float scalar, float* destination, unsigned int rows0, unsigned int cols0) {
	std::vector<std::pair<float, std::string>> results;
	size_t size = nThreads + 2ull;
	results.reserve(size);

	clock_t time;
	float avgTime;

	// GPU
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Device::multiply(source0, scalar, destination, rows0, cols0);
	}

	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "GPU " + Utils::getCUDAName()));

	// Single threaded
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Host::multiply(source0, scalar, destination, rows0, cols0, nThreads);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(1) + "T (SINGLE)"));

	//Multi threaded
	for (unsigned int threadId = 1u; threadId < nThreads + 1; threadId++) {

		time = 0;
		for (unsigned int i = 0u; i < samples; i++) {
			time += Host::multiply(source0, scalar, destination, rows0, cols0, threadId);
		}
		avgTime = static_cast<float>(time);
		avgTime /= samples;
		results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(threadId) + "T (MULTI) "));
	}

	//Print out results
	std::cout << "Multiplication (by scalar) results:\n";
	for (size_t i = 0ull; i < size; i++) {
		std::cout << results[i].second << ":\t" << results[i].first << "ms\n";
	}
	std::cout << "\n";

}

void Launcher::determinant(float* source0, unsigned int cols) {

	std::vector<std::pair<float, std::string>> results;
	size_t size = nThreads + 2ull;
	results.reserve(3);

	clock_t time;
	float avgTime;

	float dummy;

	// GPU
	/*time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Device::determinant(source0, &dummy, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "GPU " + Utils::getCUDAName()));*/

	// Single threaded
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Host::determinant(source0, &dummy, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(1) + "T (SINGLE)"));


	//Multi threaded
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Host::determinant(source0, &dummy, cols, 1);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(cols) + "T (MULTI) "));

	//Print out results
	std::cout << "Determinant results:\n";
	for (size_t i = 0ull; i < results.size(); i++) {
		std::cout << results[i].second << ":\t" << results[i].first << "ms\n";
	}
	std::cout << "\n";

}

void Launcher::transpose(float* source, float* destination, unsigned int rows, unsigned int cols) {

	std::vector<std::pair<float, std::string>> results;
	size_t size = nThreads + 2ull;
	results.reserve(size);

	clock_t time;
	float avgTime;

	// GPU
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Device::transpose(source, destination, rows, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "GPU " + Utils::getCUDAName()));
	

	// Single threaded
	time = 0;
	for (unsigned int i = 0u; i < samples; i++) {
		time += Host::transpose(source, destination, rows, cols);
	}
	avgTime = static_cast<float>(time);
	avgTime /= samples;
	results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(1) + "T (SINGLE)"));

	//Multi threaded
	for (unsigned int threadId = 1u; threadId < nThreads + 1; threadId++) {

		time = 0;
		for (unsigned int i = 0u; i < samples; i++) {
			time += time += Host::transpose(source, destination, rows, cols, threadId);
		}
		avgTime = static_cast<float>(time);
		avgTime /= samples;
		results.push_back(std::make_pair(avgTime, "CPU " + std::to_string(threadId) + "T (MULTI) "));
	}

	//Print out results
	std::cout << "Transposal results:\n";
	for (size_t i = 0ull; i < results.size(); i++) {
		std::cout << results[i].second << ":\t" << results[i].first << "ms\n";
	}
	std::cout << "\n";

}

