#pragma once

void dense_forward(
    const double* x,
    const double* w,
    const double* b,
    double* y,
    int in_dim,
    int out_dim
);

void dense_backward(
    const double* x,
    const double* w,
    const double* grad_y,
    double* grad_x,
    double* grad_w,
    double* grad_b,
    int in_dim,
    int out_dim
);

