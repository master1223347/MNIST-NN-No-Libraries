#include "math_utils.h"

// Exponential approximation
double my_exp(double x) {
    // clamp to avoid blow up in Taylor series
    if (x > 10.0) x = 10.0;
    if (x < -10.0) x = -10.0;

    double result = 1.0;
    double term = 1.0;

    // Truncated Taylor series
    for (int i = 1; i <= 10; i++) {
        term *= x / i;
        result += term;
    }

    return result;
}

// Natural logarithm approximation
double my_log(double x) {
    // clamp input to avoid log(0) and extreme gradients
    if (x < 1e-8) x = 1e-8;

    double y = 0.0; // initial guess

    for (int i = 0; i < 10; i++) {
        double ey = my_exp(y);
        y -= (ey - x) / ey;
    }

    return y;
}
