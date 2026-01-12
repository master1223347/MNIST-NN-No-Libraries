#include "dense.h"

//Forward pass
void dense_forward(
    const double* x,
    const double* w,
    const double* b,
    double* y,
    int in_dim,
    int out_dim
) { //forward pass
    for (int i = 0; i < out_dim; i++) { //loop over the output neurons
        double sum = b[i]; //add bias to each output neuron
        for (int j = 0; j < in_dim; j++) { //loop over input neurons
            sum += w[i * in_dim + j] * x[j]; //multiply each neuron by weight and add it to sum
        }
        y[i] = sum; //map each output neuron to its final value
        //y_i = b_i + sum_j (W[i,j] * x_j)
    }
}

// Backward pass
void dense_backward(
    const double* x,
    const double* w,
    const double* grad_y,
    double* grad_x,
    double* grad_w,
    double* grad_b,
    int in_dim,
    int out_dim
) { 
    // Compute grad_x = W^T * grad_y
    for (int j = 0; j < in_dim; j++) { //loop over each input neuron x_j
        grad_x[j] = 0; //Initialize sum for ∂L/∂x_j
        for (int i = 0; i < out_dim; i++) { //loop over each output neuron y_i
            //grad_x[j] += W[i,j] * grad_y[i]
            //∂L/∂x_j = sum_i ( ∂L/∂y_i * ∂y_i/∂x_j )
            //∂y_i/∂x_j = W[i,j], ∂L/∂y_i = grad_y[i]
            grad_x[j] += w[i * in_dim + j] * grad_y[i]; 
        }
    }

    //Compute grad_w = grad_y outer product x
    for (int i = 0; i < out_dim; i++) { //loop over output neurons y_i
        for (int j = 0; j < in_dim; j++) { //loop over input neurons x_j
            //grad_w[i,j] = ∂L/∂W[i,j] = ∂L/∂y_i * ∂y_i/∂W[i,j]
            //∂y_i/∂W[i,j] = x[j], ∂L/∂y_i = grad_y[i]
            //Therefore: grad_w[i,j] = grad_y[i] * x[j] (outer product)
            grad_w[i * in_dim + j] = grad_y[i] * x[j]; 
        }
    }
    //Compute grad_b
    for (int i = 0; i < out_dim; i++) {
        grad_b[i] = grad_y[i]; //copying upstream gradient directly [∂L/∂b_i = grad_y[i], bias gradient = upstream gradient]
    }

//done with FWD pass and BWD pass!!





