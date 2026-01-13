#include <pybind11/pybind11.h> //ONLY LIBRARY [I can do it manually but its a pain and doesn't contribute anything]

#include "dense.h"
#include "relu.h"
#include "loss.h"
#include "optimizer.h"

namespace py = pybind11;

//pybind stuff
PYBIND11_MODULE(neuralbinding, m) {
    m.def("dense_forward", &dense_forward);
    m.def("dense_backward", &dense_backward);

    m.def("relu_forward", &relu_forward);
    m.def("relu_backward", &relu_backward);

    m.def("softmax_ce_forward", &softmax_ce_forward);
    m.def("softmax_ce_backward", &softmax_ce_backward);

    m.def("sgd_update", &sgd_update);
    m.def("sgd_update_bias", &sgd_update_bias);
}
