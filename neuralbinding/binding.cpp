#include <pybind11/pybind11.h> //ONLY LIBRARY [I can do it manually but its a pain and doesn't contribute anything]
#include <pybind11/stl.h>

#include "dense.h"
#include "relu.h"
#include "loss.h"
#include "optimizer.h"
#include "math_utils.h"

namespace py = pybind11;

//pybind stuff
PYBIND11_MODULE(neuralbinding, m) {
    m.def("dense_forward", [](const std::vector<double>& x,
                              const std::vector<double>& w,
                              const std::vector<double>& b,
                              py::list y,
                              int in_dim,
                              int out_dim) {
        std::vector<double> y_vec = y.cast<std::vector<double>>();
        dense_forward(x, w, b, y_vec, in_dim, out_dim);
        for (int i = 0; i < out_dim; i++) {
            y[i] = y_vec[i];
        }
    });
    m.def("dense_backward", [](const std::vector<double>& x,
                               const std::vector<double>& w,
                               const std::vector<double>& grad_y,
                               py::list grad_x,
                               py::list grad_w,
                               py::list grad_b,
                               int in_dim,
                               int out_dim) {
        std::vector<double> grad_x_vec = grad_x.cast<std::vector<double>>();
        std::vector<double> grad_w_vec = grad_w.cast<std::vector<double>>();
        std::vector<double> grad_b_vec = grad_b.cast<std::vector<double>>();
        dense_backward(x, w, grad_y, grad_x_vec, grad_w_vec, grad_b_vec, in_dim, out_dim);
        for (int i = 0; i < in_dim; i++) {
            grad_x[i] = grad_x_vec[i];
        }
        for (int i = 0; i < out_dim * in_dim; i++) {
            grad_w[i] = grad_w_vec[i];
        }
        for (int i = 0; i < out_dim; i++) {
            grad_b[i] = grad_b_vec[i];
        }
    });

    m.def("relu_forward", [](py::list y, int size) {
        std::vector<double> y_vec = y.cast<std::vector<double>>();
        relu_forward(y_vec, size);
        for (int i = 0; i < size; i++) {
            y[i] = y_vec[i];
        }
    });
    m.def("relu_backward", [](const std::vector<double>& y, py::list grad_y, int size) {
        std::vector<double> grad_y_vec = grad_y.cast<std::vector<double>>();
        relu_backward(y, grad_y_vec, size);
        for (int i = 0; i < size; i++) {
            grad_y[i] = grad_y_vec[i];
        }
    });

    m.def("softmax_ce_forward", &softmax_ce_forward);
    m.def("softmax_ce_backward", [](const std::vector<double>& logits,
                                    const std::vector<double>& target,
                                    py::list grad_logits,
                                    int size) {
        std::vector<double> grad_logits_vec = grad_logits.cast<std::vector<double>>();
        softmax_ce_backward(logits, target, grad_logits_vec, size);
        for (int i = 0; i < size; i++) {
            grad_logits[i] = grad_logits_vec[i];
        }
    });

    m.def("sgd_update", [](py::list w,
                           const std::vector<double>& grad_w,
                           int size,
                           double lr) {
        std::vector<double> w_vec = w.cast<std::vector<double>>();
        sgd_update(w_vec, grad_w, size, lr);
        for (int i = 0; i < size; i++) {
            w[i] = w_vec[i];
        }
    });
    m.def("sgd_update_bias", [](py::list b,
                                const std::vector<double>& grad_b,
                                int size,
                                double lr) {
        std::vector<double> b_vec = b.cast<std::vector<double>>();
        sgd_update_bias(b_vec, grad_b, size, lr);
        for (int i = 0; i < size; i++) {
            b[i] = b_vec[i];
        }
    });
}