#include "math_utils.h"

// Exponential approximation without cmath
double my_exp(double x) {
    // Clamp input to prevent overflow
    if (x > 88.0) x = 88.0;   // e^88 ~ max for float/double in NN
    if (x < -88.0) x = -88.0;

    // Range reduction: find n such that x = n*ln2 + f, -0.5*ln2 <= f <= 0.5*ln2
    const double ln2 = 0.69314718056;
    double n = (int)(x / ln2 + (x >= 0 ? 0.5 : -0.5)); // nearest integer
    double f = x - n * ln2;

    // Polynomial approximation for exp(f) using Horner's method
    double expf = 1.0 + f*(1.0 + f*(0.5 + f*(0.1666666667 + f*(0.0416666667 + f*0.0083333333))));

    // Compute 2^n by repeated multiplication (no ldexp)
    double pow2 = 1.0;
    if (n > 0) {
        for (int i = 0; i < (int)n; i++) pow2 *= 2.0;
    } else if (n < 0) {
        for (int i = 0; i < -(int)n; i++) pow2 *= 0.5;
    }

    return expf * pow2;
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
