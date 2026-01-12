#pragma once

double mse_forward(
    const double* y,
    const double* t,
    int size
);

void mse_backward(
    const double* y,
    const double* t,
    double* grad_y,
    int size
);
