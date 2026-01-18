#include "loss.h"
#include "math_utils.h"

// Forward pass: softmax + cross entropy
double softmax_ce_forward(
    const std::vector<double>& logits,
    const std::vector<double>& target,
    int size
) {
    std::vector<double> softmax_out(10); // MNIST output size
    double sum_exp = 0.0;

    // compute max for numeric stability (softmax shift)
    double max_logit = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // 1. compute exponentials with clamp
    for (int i = 0; i < size; i++) {
        double z = logits[i];
        // apply softmax shift instead of clamping
        z = z - max_logit;
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
    const std::vector<double>& logits,
    const std::vector<double>& target,
    std::vector<double>& grad_logits,
    int size
) {
    std::vector<double> softmax_out(10);
    double sum_exp = 0.0;

    // compute max for numeric stability (softmax shift)
    double max_logit = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // 1. compute exponentials with clamp
    for (int i = 0; i < size; i++) {
        double z = logits[i];
        // apply softmax shift instead of clamping
        z = z - max_logit;
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
