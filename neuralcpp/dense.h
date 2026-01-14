#pragma once
#include <vector>  // required for std::vector

void dense_forward(
    const std::vector<double>& x,
    const std::vector<double>& w,
    const std::vector<double>& b,
    std::vector<double>& y,
    int in_dim,
    int out_dim
);

void dense_backward(
    const std::vector<double>& x,
    const std::vector<double>& w,
    const std::vector<double>& grad_y,
    std::vector<double>& grad_x,
    std::vector<double>& grad_w,
    std::vector<double>& grad_b,
    int in_dim,
    int out_dim
);