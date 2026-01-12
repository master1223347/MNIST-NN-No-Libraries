#include <iostream>
#include "dense.h"

int main() {
    double x[3] = {1.0, 2.0, 3.0};

    // 2 outputs, 3 inputs
    double w[6] = {
        1, 0, 0,
        0, 1, 0
    };

    double b[2] = {0.0, 0.0};
    double y[2];

    dense_forward(x, w, b, y, 3, 2);

    std::cout << y[0] << " " << y[1] << std::endl;
}
