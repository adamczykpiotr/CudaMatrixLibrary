#pragma once
#include <string>
#include <vector>
#include "Host.h"
#include "Device.h"
#include "Utils.h"

class Launcher {

	static size_t nThreads;
	static unsigned int samples;
public:
	static void setNThreads(size_t nThreads_);
	static void setSamples(unsigned int samples_);

	static void add(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols);
	static void substract(float* source0, float* source1, float* destination, unsigned int rows, unsigned int cols);
	static void multiply(float* source0, float* source1, float* destination, unsigned int rows0, unsigned int cols0, unsigned int cols1);
	static void multiply(float* source0, float scalar, float* destination, unsigned int rows0, unsigned int cols0);
	static void determinant(float* source, unsigned int size);
	static void transpose(float * source, float * destination, unsigned int rows, unsigned int cols);

};

