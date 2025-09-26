#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include <random>
#include <iomanip>

#include "headers/Complex.h"
#include "headers/Matrix.h"
#include "headers/MLP.h"

using namespace std;

template<typename T>
struct MLPExperimentTypes {
    struct DataSample {
        vector<T> inputs;
        vector<T> outputs;
    };

    struct Dataset {
        vector<DataSample> samples;
        string name;
        int inputDim;
        int outputDim;
    };

    struct HyperparameterConfig {
        vector<int> architecture;
        T learningRate;
        int epochs;
        string description;
    };

    struct ExperimentResult {
        HyperparameterConfig config;
        T trainLoss;
        T testLoss;
        T trainAccuracy;
        T testAccuracy;
        T splitRatio;
    };
};

template<typename T>
typename MLPExperimentTypes<T>::Dataset loadDatasetFromCSV(string filename, string datasetName) {
    typename MLPExperimentTypes<T>::Dataset dataset;
    dataset.name = datasetName;
    
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return dataset;
    }
    
    string line;
    bool isHeader = true;
    
    while (getline(file, line)) {
        if (isHeader) {
            isHeader = false;
            continue;
        }
        
        stringstream ss(line);
        string value;
        vector<T> values;
        
        while (getline(ss, value, ',')) {
            values.push_back(T(stod(value)));
        }
        
        typename MLPExperimentTypes<T>::DataSample sample;
        if (datasetName == "XOR") {
            sample.inputs = {values[0], values[1]};
            sample.outputs = {values[2]};
            dataset.inputDim = 2;
            dataset.outputDim = 1;
        } else if (datasetName == "Binary Adder") {
            sample.inputs = {values[0], values[1], values[2], values[3], values[4]};
            sample.outputs = {values[5], values[6], values[7]};
            dataset.inputDim = 5;
            dataset.outputDim = 3;
        }
        
        dataset.samples.push_back(sample);
    }
    
    file.close();
    return dataset;
}

template<typename T>
pair<typename MLPExperimentTypes<T>::Dataset, typename MLPExperimentTypes<T>::Dataset> splitDataset(const typename MLPExperimentTypes<T>::Dataset& original, T trainRatio, int seed = 42) {
    typename MLPExperimentTypes<T>::Dataset trainSet, testSet;
    trainSet.name = original.name + " (Train)";
    testSet.name = original.name + " (Test)";
    trainSet.inputDim = original.inputDim;
    trainSet.outputDim = original.outputDim;
    testSet.inputDim = original.inputDim;
    testSet.outputDim = original.outputDim;
    
    vector<int> indices(original.samples.size());
    iota(indices.begin(), indices.end(), 0);
    
    mt19937 rng(seed);
    shuffle(indices.begin(), indices.end(), rng);
    
    int trainSize = static_cast<int>(original.samples.size() * trainRatio);
    
    for (int i = 0; i < trainSize; ++i) {
        trainSet.samples.push_back(original.samples[indices[i]]);
    }
    
    for (int i = trainSize; i < static_cast<int>(indices.size()); ++i) {
        testSet.samples.push_back(original.samples[indices[i]]);
    }
    
    return make_pair(trainSet, testSet);
}

template<typename T>
pair<vector<Matrix<T>>, vector<Matrix<T>>> datasetToMatrices(const typename MLPExperimentTypes<T>::Dataset& dataset) {
    vector<Matrix<T>> inputs, outputs;
    
    for (auto sample : dataset.samples) {
        inputs.push_back(Matrix<T>(sample.inputs, true));
        outputs.push_back(Matrix<T>(sample.outputs, true));
    }
    
    return make_pair(inputs, outputs);
}

template<typename T>
typename MLPExperimentTypes<T>::ExperimentResult runExperiment(const typename MLPExperimentTypes<T>::Dataset& trainSet, const typename MLPExperimentTypes<T>::Dataset& testSet, const typename MLPExperimentTypes<T>::HyperparameterConfig& config, T splitRatio) {
    typename MLPExperimentTypes<T>::ExperimentResult result;
    result.config = config;
    result.splitRatio = splitRatio;
    
    auto [trainInputs, trainTargets] = datasetToMatrices<T>(trainSet);
    auto [testInputs, testTargets] = datasetToMatrices<T>(testSet);
    
    MLP<T> mlp(config.architecture, config.learningRate);
    
    mlp.trainWithValidation(trainInputs, trainTargets, testInputs, testTargets, config.epochs, false);
    
    result.trainLoss = mlp.evaluate(trainInputs, trainTargets);
    result.testLoss = mlp.evaluate(testInputs, testTargets);
    result.trainAccuracy = mlp.calculateAccuracy(trainInputs, trainTargets);
    result.testAccuracy = mlp.calculateAccuracy(testInputs, testTargets);
    
    return result;
}

template<typename T>
void printResults(vector<typename MLPExperimentTypes<T>::ExperimentResult> results, string datasetName) {
    cout << "\n" << string(120, '=') << endl;
    cout << "EXPERIMENT RESULTS FOR " << datasetName << " DATASET" << endl;
    cout << string(120, '=') << endl;
    
    cout << left << setw(25) << "Architecture" << setw(10) << "LR" << setw(8) << "Epochs" << setw(8) << "Split" << setw(12) << "Train Loss" << setw(12) << "Test Loss" << setw(12) << "Train Acc" << setw(12) << "Test Acc" << setw(15) << "Description" << endl;
    cout << string(120, '-') << endl;
    
    for (auto result : results) {
        string archStr = "";
        for (size_t i = 0; i < result.config.architecture.size(); ++i) {
            archStr += to_string(result.config.architecture[i]);
            if (i < result.config.architecture.size() - 1) archStr += "-";
        }
        
        cout << left << setw(25) << archStr << setw(10) << fixed << setprecision(3) << result.config.learningRate << setw(8) << result.config.epochs << setw(8) << fixed << setprecision(2) << result.splitRatio << setw(12) << fixed << setprecision(4) << result.trainLoss << setw(12) << fixed << setprecision(4) << result.testLoss << setw(12) << fixed << setprecision(3) << result.trainAccuracy << setw(12) << fixed << setprecision(3) << result.testAccuracy << setw(15) << result.config.description << endl;
    }
    cout << string(120, '-') << endl;
}

template<typename T>
void printBestConfigurations(vector<typename MLPExperimentTypes<T>::ExperimentResult> results, string datasetName) {
    cout << "\nBEST CONFIGURATIONS FOR " << datasetName << ":" << endl;
    cout << string(60, '-') << endl;
    
    auto bestAccuracy = *max_element(results.begin(), results.end(), [](typename MLPExperimentTypes<T>::ExperimentResult a, typename MLPExperimentTypes<T>::ExperimentResult b) {
        return a.testAccuracy < b.testAccuracy;
    });

    auto bestLoss = *min_element(results.begin(), results.end(), [](typename MLPExperimentTypes<T>::ExperimentResult a, typename MLPExperimentTypes<T>::ExperimentResult b) {
        return a.testLoss < b.testLoss;
    });

    cout << "Best Test Accuracy: " << fixed << setprecision(3) << bestAccuracy.testAccuracy << " (" << bestAccuracy.config.description << ")" << endl;
    cout << "Best Test Loss: " << fixed << setprecision(4) << bestLoss.testLoss  << " (" << bestLoss.config.description << ")" << endl;
}

int main() {
    using ExpTypes = MLPExperimentTypes<double>;
    using Dataset = ExpTypes::Dataset;
    using HyperparameterConfig = ExpTypes::HyperparameterConfig;
    using ExperimentResult = ExpTypes::ExperimentResult;
    
    cout << string(80, '=') << endl;
    cout << "MULTILAYER PERCEPTRON COMPREHENSIVE EXPERIMENT SUITE" << endl;
    cout << string(80, '=') << endl;
    
    cout << "\n[1] Loading Datasets..." << endl;
    Dataset xorDataset = loadDatasetFromCSV<double>("datasets/xor_dataset.csv", "XOR");
    Dataset adderDataset = loadDatasetFromCSV<double>("datasets/binary_adder_dataset.csv", "Binary Adder");
    
    cout << "✓ XOR Dataset: " << xorDataset.samples.size() << " samples, " << xorDataset.inputDim << " inputs, " << xorDataset.outputDim << " outputs" << endl;
    cout << "✓ Binary Adder Dataset: " << adderDataset.samples.size() << " samples, " << adderDataset.inputDim << " inputs, " << adderDataset.outputDim << " outputs" << endl;
    
    cout << "\n[2] Defining Hyperparameter Configurations..." << endl;
    
    vector<HyperparameterConfig> xorConfigs = {
        {{2, 4, 1}, 0.5, 1000, "Small Hidden"},
        {{2, 8, 1}, 0.5, 1000, "Medium Hidden"},
        {{2, 16, 1}, 0.3, 1000, "Large Hidden"},
        {{2, 4, 4, 1}, 0.3, 1500, "Two Hidden Small"},
        {{2, 8, 4, 1}, 0.2, 1500, "Two Hidden Medium"},
        
        {{2, 8, 1}, 0.1, 1000, "Low LR"},
        {{2, 8, 1}, 0.3, 1000, "Medium LR"},
        {{2, 8, 1}, 0.7, 1000, "High LR"},
        
        {{2, 8, 1}, 0.5, 500, "Short Training"},
        {{2, 8, 1}, 0.5, 2000, "Long Training"},
    };
    
    vector<HyperparameterConfig> adderConfigs = {
        {{5, 8, 3}, 0.3, 1000, "Small Hidden"},
        {{5, 16, 3}, 0.3, 1000, "Medium Hidden"},
        {{5, 32, 3}, 0.2, 1000, "Large Hidden"},
        {{5, 10, 8, 3}, 0.2, 1500, "Two Hidden Small"},
        {{5, 16, 8, 3}, 0.15, 1500, "Two Hidden Medium"},
        {{5, 20, 10, 3}, 0.1, 2000, "Two Hidden Large"},
    
        {{5, 16, 3}, 0.1, 1000, "Low LR"},
        {{5, 16, 3}, 0.5, 1000, "High LR"},
        
        {{5, 16, 3}, 0.3, 500, "Short Training"},
        {{5, 16, 3}, 0.3, 2000, "Long Training"},
    };
    
    cout << "✓ Defined " << xorConfigs.size() << " configurations for XOR" << endl;
    cout << "✓ Defined " << adderConfigs.size() << " configurations for Binary Adder" << endl;
    
    vector<double> splitRatios = {0.5, 0.7, 0.8};
    
    cout << "\n[3] Choose Configuration for XOR Experiments..." << endl;
    
    // Display available configurations
    cout << "\nAvailable XOR Configurations:" << endl;
    for (size_t i = 0; i < xorConfigs.size(); ++i) {
        cout << i + 1 << ". " << xorConfigs[i].description << " (";
        for (size_t j = 0; j < xorConfigs[i].architecture.size(); ++j) {
            cout << xorConfigs[i].architecture[j];
            if (j < xorConfigs[i].architecture.size() - 1) cout << "-";
        }
        cout << ", LR=" << xorConfigs[i].learningRate << ", Epochs=" << xorConfigs[i].epochs << ")" << endl;
    }
    
    int xorChoice;
    cout << "\nEnter your choice (1-" << xorConfigs.size() << "): ";
    cin >> xorChoice;
    
    if (xorChoice < 1 || xorChoice > static_cast<int>(xorConfigs.size())) {
        cout << "Invalid choice! Using first configuration." << endl;
        xorChoice = 1;
    }
    xorChoice--; // Convert to 0-based index
    
    // Display available split ratios
    cout << "\nAvailable Split Ratios:" << endl;
    for (size_t i = 0; i < splitRatios.size(); ++i) {
        cout << i + 1 << ". " << splitRatios[i] << " (Train: " << splitRatios[i] * 100 << "%, Test: " << (1 - splitRatios[i]) * 100 << "%)" << endl;
    }
    
    int splitChoice;
    cout << "\nEnter split ratio choice (1-" << splitRatios.size() << "): ";
    cin >> splitChoice;
    
    if (splitChoice < 1 || splitChoice > static_cast<int>(splitRatios.size())) {
        cout << "Invalid choice! Using first split ratio." << endl;
        splitChoice = 1;
    }
    splitChoice--; // Convert to 0-based index
    
    vector<ExperimentResult> xorResults;
    cout << "\nRunning XOR experiment with:" << endl;
    cout << "Config: " << xorConfigs[xorChoice].description << endl;
    cout << "Split: " << splitRatios[splitChoice] << endl;
    
    auto [xorTrain, xorTest] = splitDataset<double>(xorDataset, splitRatios[splitChoice]);
    ExperimentResult result = runExperiment<double>(xorTrain, xorTest, xorConfigs[xorChoice], splitRatios[splitChoice]);
    xorResults.push_back(result);
    
    cout << "\n[4] Choose Configuration for Binary Adder Experiments..." << endl;
    
    // Display available configurations
    cout << "\nAvailable Binary Adder Configurations:" << endl;
    for (size_t i = 0; i < adderConfigs.size(); ++i) {
        cout << i + 1 << ". " << adderConfigs[i].description << " (";
        for (size_t j = 0; j < adderConfigs[i].architecture.size(); ++j) {
            cout << adderConfigs[i].architecture[j];
            if (j < adderConfigs[i].architecture.size() - 1) cout << "-";
        }
        cout << ", LR=" << adderConfigs[i].learningRate << ", Epochs=" << adderConfigs[i].epochs << ")" << endl;
    }
    
    int adderChoice;
    cout << "\nEnter your choice (1-" << adderConfigs.size() << "): ";
    cin >> adderChoice;
    
    if (adderChoice < 1 || adderChoice > static_cast<int>(adderConfigs.size())) {
        cout << "Invalid choice! Using first configuration." << endl;
        adderChoice = 1;
    }
    adderChoice--; 
    
    cout << "\nAvailable Split Ratios:" << endl;
    for (size_t i = 0; i < splitRatios.size(); ++i) {
        cout << i + 1 << ". " << splitRatios[i] << " (Train: " << splitRatios[i] * 100 << "%, Test: " << (1 - splitRatios[i]) * 100 << "%)" << endl;
    }
    
    int adderSplitChoice;
    cout << "\nEnter split ratio choice (1-" << splitRatios.size() << "): ";
    cin >> adderSplitChoice;
    
    if (adderSplitChoice < 1 || adderSplitChoice > static_cast<int>(splitRatios.size())) {
        cout << "Invalid choice! Using first split ratio." << endl;
        adderSplitChoice = 1;
    }
    adderSplitChoice--;
    
    vector<ExperimentResult> adderResults;
    cout << "\nRunning Binary Adder experiment with:" << endl;
    cout << "Config: " << adderConfigs[adderChoice].description << endl;
    cout << "Split: " << splitRatios[adderSplitChoice] << endl;
    
    auto [adderTrain, adderTest] = splitDataset<double>(adderDataset, splitRatios[adderSplitChoice]);
    ExperimentResult adderResult = runExperiment<double>(adderTrain, adderTest, adderConfigs[adderChoice], splitRatios[adderSplitChoice]);
    adderResults.push_back(adderResult);
    
    cout << "\n[5] Results Analysis..." << endl;
    
    printResults<double>(xorResults, "XOR");
    printBestConfigurations<double>(xorResults, "XOR");
    
    printResults<double>(adderResults, "BINARY ADDER");
    printBestConfigurations<double>(adderResults, "BINARY ADDER");
    
    cout << "\n" << string(80, '=') << endl;
    cout << "EXPERIMENT SUMMARY" << endl;
    cout << string(80, '=') << endl;
    cout << "Total experiments conducted: " << (xorResults.size() + adderResults.size()) << endl;
    cout << "XOR Configuration: " << xorResults[0].config.description << endl;
    cout << "Binary Adder Configuration: " << adderResults[0].config.description << endl;
    
    cout << "\nUser-Selected Results:" << endl;
    cout << "• XOR Test Accuracy: " << fixed << setprecision(3) << xorResults[0].testAccuracy << endl;
    cout << "• XOR Test Loss: " << fixed << setprecision(4) << xorResults[0].testLoss << endl;
    cout << "• Binary Adder Test Accuracy: " << fixed << setprecision(3) << adderResults[0].testAccuracy << endl;
    cout << "• Binary Adder Test Loss: " << fixed << setprecision(4) << adderResults[0].testLoss << endl;
    
    cout << "\n" << string(80, '=') << endl;
    cout << "EXPERIMENT COMPLETED SUCCESSFULLY!" << endl;
    cout << string(80, '=') << endl;
    
    return 0;
}