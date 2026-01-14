#pragma once
#include <vector>  // required for std::vector

double softmax_ce_forward(
    const std::vector<double>& logits,
    const std::vector<double>& target,
    int size
);

void softmax_ce_backward(
    const std::vector<double>& logits,
    const std::vector<double>& target,
    std::vector<double>& grad_logits,
    int size
);
