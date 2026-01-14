#include "relu.h"

//ReLu forward pass
void relu_forward(std::vector<double>& y, int size) {
    for (int i = 0; i < size; i++) { //loop over each output neuron
        if (y[i] < 0.0) y[i] = 0.0; //ReLu activation; if < 0, output 0
    }
}

//Relu backward pass
void relu_backward(const std::vector<double>& y, std::vector<double>& grad_y, int size) {
    for (int i = 0; i < size; i++) { //loop over each output neuron
        if (y[i] <= 0.0) grad_y[i] = 0.0; //sets ReLu derivative to 0
    }
}
