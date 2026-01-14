#include "optimizer.h"

void sgd_update(
    std::vector<double>& w,
    const std::vector<double>& grad_w,
    int size,
    double lr
) {
    for (int i = 0; i < size; i++) { //loops over all weight params
        w[i] -= lr * grad_w[i]; //gradient descent
    }
}

void sgd_update_bias(
    std::vector<double>& b,
    const std::vector<double>& grad_b,
    int size,
    double lr
) {
    for (int i = 0; i < size; i++) { //loops over every bias
        b[i] -= lr * grad_b[i]; //gradient descent
    }
}
