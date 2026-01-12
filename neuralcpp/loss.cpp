#include "loss.h"

double mse_forward(
    const double* y,
    const double* t,
    int size
) {
    double loss = 0.0;

    for (int i = 0; i < size; i++) {
        double diff = y[i] - t[i];
        loss += diff * diff;
    }

    return loss / size;
}

void mse_backward(
    const double* y,
    const double* t,
    double* grad_y,
    int size
) {
    for (int i = 0; i < size; i++) {
        grad_y[i] = 2.0 * (y[i] - t[i]) / size;
    }
}