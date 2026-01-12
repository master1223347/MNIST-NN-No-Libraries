#pragma once

void relu_forward(double* y, int size);
void relu_backward(const double* y, double* grad_y, int size);