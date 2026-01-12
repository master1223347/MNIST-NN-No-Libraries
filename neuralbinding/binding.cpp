#include <pybind11/pybind11.h>

#include "dense.h"
#include "relu.h"
#include "loss.h"
#include "optimizer.h"

namespace py = pybind11;

PYBIND11_MODULE(neuralbinding, m) {
    m.def("dense_forward", &dense_forward);
    m.def("dense_backward", &dense_backward);

    m.def("relu_forward", &relu_forward);
    m.def("relu_backward", &relu_backward);

    m.def("mse_forward", &mse_forward);
    m.def("mse_backward", &mse_backward);

    m.def("sgd_update", &sgd_update);
    m.def("sgd_update_bias", &sgd_update_bias);
}
