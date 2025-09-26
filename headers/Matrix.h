#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>

#include "Complex.h"

using namespace std;

template<typename T>
class Matrix {
private:
    vector<vector<T>> data;
    size_t rows, cols;

public:
    Matrix() : rows(0), cols(0) {}
    
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, vector<T>(cols, T{}));
    }

    Matrix(vector<vector<T>> input) : data(input) {
        rows = data.size();
        cols = rows > 0 ? data[0].size() : 0;
    }

    Matrix(vector<T> vec, bool column = true) {
        if (column) {
            rows = vec.size();
            cols = 1;
            data.resize(rows, vector<T>(1));
            for (size_t i = 0; i < rows; ++i) {
                data[i][0] = vec[i];
            }
        } else {
            rows = 1;
            cols = vec.size();
            data.resize(1, vec);
        }
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    T operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols) {
            throw out_of_range("Matrix index out of bounds");
        }
        return data[row][col];
    }

    void set(size_t row, size_t col, T value) {
        if (row >= rows || col >= cols) {
            throw out_of_range("Matrix index out of bounds");
        }

        data[row][col] = value;
    }

    Matrix operator+(Matrix other) const {
        if (rows != other.rows || cols != other.cols) {
            throw invalid_argument("Matrix dimensions must match for addition");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.set(i, j, data[i][j] + other(i, j));
            }
        }
        return result;
    }

    Matrix operator-(Matrix other) const {
        if (rows != other.rows || cols != other.cols) {
            throw invalid_argument("Matrix dimensions must match for subtraction");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.set(i, j, data[i][j] - other(i, j));
            }
        }
        return result;
    }

    Matrix operator*(Matrix other) const {
        if (cols != other.rows) {
            throw invalid_argument("Invalid matrix dimensions for multiplication");
        }
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                T sum = T{};
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    Matrix operator*(T scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.set(i, j, data[i][j] * scalar);
            }
        }
        return result;
    }

    void randomize(T minVal = T{-1}, T maxVal = T{1}) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis{double(minVal), double(maxVal)};
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = T(dis(gen));
            }
        }
    }

    vector<T> toVector() const {
        vector<T> result;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.push_back(data[i][j]);
            }
        }
        return result;
    }

    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }
};

#endif