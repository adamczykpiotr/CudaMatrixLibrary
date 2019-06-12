#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <vector>


class Utils {
public:
	static void validateCUDA();
	static std::string getCUDAName(int deviceId = 0);
	static void print(float* matrix, unsigned int rows, unsigned int cols);
};