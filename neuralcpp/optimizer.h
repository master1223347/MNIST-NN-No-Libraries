#pragma once
#include <vector>  // required for std::vector

void sgd_update(
    std::vector<double>& w,
    const std::vector<double>& grad_w,
    int size,
    double lr
);

void sgd_update_bias(
    std::vector<double>& b,
    const std::vector<double>& grad_b,
    int size,
    double lr
);
