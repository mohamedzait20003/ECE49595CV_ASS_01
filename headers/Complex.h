#ifndef COMPLEX_H
#define COMPLEX_H

#include <iostream>
#include <cmath>

using namespace std;

template<typename T>
class ComplexNumber {
    T real;
    T imag;

    public:
    ComplexNumber(T r = T{}, T i = T{}) : real(r), imag(i) {}

    T getReal() const {
        return real;
    }

    T getImag() const {
        return imag;
    }

    T getMagnitude() const {
        return sqrt(real * real + imag * imag);
    }

    ComplexNumber operator+(ComplexNumber other) const {
        return ComplexNumber(real + other.real, imag + other.imag);
    }

    ComplexNumber operator-(ComplexNumber other) const {
        return ComplexNumber(real - other.real, imag - other.imag);
    }

    ComplexNumber operator*(ComplexNumber other) const {
        T newReal = real * other.real - imag * other.imag;
        T newImag = real * other.imag + imag * other.real;
        return ComplexNumber(newReal, newImag);
    }

    ComplexNumber operator*(T scalar) const {
        return ComplexNumber(real * scalar, imag * scalar);
    }

    ComplexNumber operator+=(ComplexNumber other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    ComplexNumber operator-=(ComplexNumber other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    friend ostream operator<<(ostream os, ComplexNumber c) {
        os << c.real;
        if (c.imag >= 0) os << "+";
        os << c.imag << "i";
        return os;
    }
};

#endif // COMPLEX_H