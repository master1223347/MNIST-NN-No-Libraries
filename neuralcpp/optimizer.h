#pragma once

void sgd_update(
    double* w,
    const double* grad_w,
    int size,
    double lr
);

void sgd_update_bias(
    double* b,
    const double* grad_b,
    int size,
    double lr
);