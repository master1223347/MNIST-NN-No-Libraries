#pragma once
#include <vector>  // required for std::vector

void relu_forward(std::vector<double>& y, int size);
void relu_backward(const std::vector<double>& y, std::vector<double>& grad_y, int size);
