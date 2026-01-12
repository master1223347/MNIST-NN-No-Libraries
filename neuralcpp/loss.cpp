#include "loss.h"

//Mean Squared Error forward pass
double mse_forward(
    const double* y,
    const double* t,
    int size
) {
    double loss = 0.0; //store error

    for (int i = 0; i < size; i++) { //looping over all output neurons
        double diff = y[i] - t[i]; //calculates diff between model pred and actual answer
        loss += diff * diff; //squares diffs
    }

    return loss / size; //avg error
}

//Mean Squared Error backward pass
void mse_backward(
    const double* y,
    const double* t,
    double* grad_y,
    int size
) {
    for (int i = 0; i < size; i++) { //loops over all output neurons
        grad_y[i] = 2.0 * (y[i] - t[i]) / size; //loss gradient
    }
}