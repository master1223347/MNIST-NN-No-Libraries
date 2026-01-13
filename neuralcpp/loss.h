#pragma once

double softmax_ce_forward(
    const double* logits,
    const double* target,
    int size
);

void softmax_ce_backward(
    const double* logits,
    const double* target,
    double* grad_logits,
    int size
);