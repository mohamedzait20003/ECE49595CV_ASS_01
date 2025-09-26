#ifndef MLP_H
#define MLP_H

#include <cmath>
#include <vector>
#include <iostream>

#include "Matrix.h"

using namespace std;

template<typename T>
class MLP {
private:
    vector<Matrix<T>> weights;
    vector<Matrix<T>> biases;
    vector<int> layerSizes;
    T learningRate;

    Matrix<T> sigmoid(Matrix<T> input) {
        Matrix<T> result(input.getRows(), input.getCols());
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                result.set(i, j, T(1.0) / (T(1.0) + T(exp(-double(input(i, j))))));
            }
        }
        return result;
    }

    Matrix<T> sigmoidDerivative(Matrix<T> sigmoid_output) {
        Matrix<T> result(sigmoid_output.getRows(), sigmoid_output.getCols());
        for (size_t i = 0; i < sigmoid_output.getRows(); ++i) {
            for (size_t j = 0; j < sigmoid_output.getCols(); ++j) {
                T s = sigmoid_output(i, j);
                result.set(i, j, s * (T(1.0) - s));
            }
        }
        return result;
    }

public:
    MLP(vector<int> layers, T lr = T(0.01)) : layerSizes(layers), learningRate(lr) {
        
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            Matrix<T> w(layers[i + 1], layers[i]);
            Matrix<T> b(layers[i + 1], 1);
            
            w.randomize();
            b.randomize();
            
            weights.push_back(w);
            biases.push_back(b);
        }
    }

    Matrix<T> forward(Matrix<T> input) {
        Matrix<T> activation = input;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            Matrix<T> linear = weights[i] * activation + biases[i];
            activation = sigmoid(linear);
        }
        
        return activation;
    }

    void trainWithValidation(vector<Matrix<T>> trainInputs, vector<Matrix<T>> trainTargets, vector<Matrix<T>> valInputs, vector<Matrix<T>> valTargets, int epochs, bool verbose = true) {
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            T totalLoss = T{};
            
            vector<Matrix<T>> weightGradients;
            vector<Matrix<T>> biasGradients;
            
            for (size_t i = 0; i < weights.size(); ++i) {
                weightGradients.push_back(Matrix<T>(weights[i].getRows(), weights[i].getCols()));
                biasGradients.push_back(Matrix<T>(biases[i].getRows(), biases[i].getCols()));
            }
            
            for (size_t sample = 0; sample < trainInputs.size(); ++sample) {
                vector<Matrix<T>> activations;
                vector<Matrix<T>> linearOutputs;
                
                Matrix<T> activation = trainInputs[sample];
                activations.push_back(activation);
                
                for (size_t i = 0; i < weights.size(); ++i) {
                    Matrix<T> linear = weights[i] * activation + biases[i];
                    linearOutputs.push_back(linear);
                    activation = sigmoid(linear);
                    activations.push_back(activation);
                }
                
                Matrix<T> error = activations.back() - trainTargets[sample];
                for (size_t i = 0; i < error.getRows(); ++i) {
                    for (size_t j = 0; j < error.getCols(); ++j) {
                        totalLoss += error(i, j) * error(i, j);
                    }
                }
                
                vector<Matrix<T>> deltas(weights.size());
                
                Matrix<T> outputSigmoidDeriv = sigmoidDerivative(activations.back());
                deltas[deltas.size() - 1] = Matrix<T>(error.getRows(), error.getCols());
                for (size_t i = 0; i < error.getRows(); ++i) {
                    for (size_t j = 0; j < error.getCols(); ++j) {
                        deltas[deltas.size() - 1].set(i, j, T(2.0) * error(i, j) * outputSigmoidDeriv(i, j));
                    }
                }
                
                for (int i = weights.size() - 2; i >= 0; --i) {
                    Matrix<T> sigmoidDeriv = sigmoidDerivative(activations[i + 1]);
                    deltas[i] = Matrix<T>(weights[i + 1].getCols(), 1);
                    
                    for (size_t j = 0; j < deltas[i].getRows(); ++j) {
                        T sum = T{};
                        for (size_t k = 0; k < weights[i + 1].getRows(); ++k) {
                            sum += weights[i + 1](k, j) * deltas[i + 1](k, 0);
                        }
                        deltas[i].set(j, 0, sum * sigmoidDeriv(j, 0));
                    }
                }
                
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].getRows(); ++j) {
                        for (size_t k = 0; k < weights[i].getCols(); ++k) {
                            T gradient = deltas[i](j, 0) * activations[i](k, 0);
                            weightGradients[i].set(j, k, weightGradients[i](j, k) + gradient);
                        }
                        biasGradients[i].set(j, 0, biasGradients[i](j, 0) + deltas[i](j, 0));
                    }
                }
            }
            
            T sampleCount = T(trainInputs.size());
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[i].getRows(); ++j) {
                    for (size_t k = 0; k < weights[i].getCols(); ++k) {
                        T avgGradient = weightGradients[i](j, k) * (T(1.0) / sampleCount);
                        weights[i].set(j, k, weights[i](j, k) - learningRate * avgGradient);
                    }
                    T avgBiasGradient = biasGradients[i](j, 0) * (T(1.0) / sampleCount);
                    biases[i].set(j, 0, biases[i](j, 0) - learningRate * avgBiasGradient);
                }
            }
            
            if (verbose && epoch % 100 == 0) {
                T trainLoss = totalLoss * (T(1.0) / sampleCount);
                T valLoss = evaluate(valInputs, valTargets);
                cout << "Epoch " << epoch << " - Train Loss: " << trainLoss << ", Val Loss: " << valLoss << endl;
            }
        }
    }

    T evaluate(vector<Matrix<T>> testInputs, vector<Matrix<T>> testTargets) {
        T totalLoss = T{};
        
        for (size_t sample = 0; sample < testInputs.size(); ++sample) {
            Matrix<T> output = forward(testInputs[sample]);
            Matrix<T> error = output - testTargets[sample];
            
            for (size_t i = 0; i < error.getRows(); ++i) {
                for (size_t j = 0; j < error.getCols(); ++j) {
                    totalLoss += error(i, j) * error(i, j);
                }
            }
        }
        
        return totalLoss * (T(1.0) / T(testInputs.size()));
    }

    T calculateAccuracy(vector<Matrix<T>> testInputs, vector<Matrix<T>> testTargets, T threshold = T(0.5)) {
        T correct = T{};
        
        for (size_t sample = 0; sample < testInputs.size(); ++sample) {
            Matrix<T> output = forward(testInputs[sample]);
            bool allCorrect = true;
            
            for (size_t i = 0; i < output.getRows(); ++i) {
                T predicted = output(i, 0) > threshold ? T(1.0) : T(0.0);
                T target = testTargets[sample](i, 0);
                if (abs(predicted - target) > T(0.1)) {
                    allCorrect = false;
                    break;
                }
            }
            
            if (allCorrect) {
                correct += T(1.0);
            }
        }
        
        return correct * (T(1.0) / T(testInputs.size()));
    }
};

#endif