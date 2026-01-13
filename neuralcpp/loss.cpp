#include "loss.h"
#include "math_utils.h"

// Forward pass: softmax + cross entropy
double softmax_ce_forward(
    const double* logits,
    const double* target,
    int size
) {
    double softmax_out[10]; // MNIST output size
    double sum_exp = 0.0;

    // 1. compute exponentials with clamp
    for (int i = 0; i < size; i++) {
        double z = logits[i];
        if (z > 5.0) z = 5.0;      // clamp high
        if (z < -5.0) z = -5.0;    // clamp low
        softmax_out[i] = my_exp(z);
        sum_exp += softmax_out[i];
    }

    // 2. normalize softmax
    for (int i = 0; i < size; i++) {
        softmax_out[i] /= sum_exp;
    }

    // 3. compute cross entropy loss
    double loss = 0.0;
    for (int i = 0; i < size; i++) {
        loss -= target[i] * my_log(softmax_out[i]);
    }

    return loss;
}

// Backward pass: gradient w.r.t logits
void softmax_ce_backward(
    const double* logits,
    const double* target,
    double* grad_logits,
    int size
) {
    double softmax_out[10];
    double sum_exp = 0.0;

    // 1. compute exponentials with clamp
    for (int i = 0; i < size; i++) {
        double z = logits[i];
        if (z > 5.0) z = 5.0;
        if (z < -5.0) z = -5.0;
        softmax_out[i] = my_exp(z);
        sum_exp += softmax_out[i];
    }

    // 2. normalize softmax
    for (int i = 0; i < size; i++) {
        softmax_out[i] /= sum_exp;
    }

    // 3. compute gradient
    for (int i = 0; i < size; i++) {
        grad_logits[i] = softmax_out[i] - target[i];
    }
}
