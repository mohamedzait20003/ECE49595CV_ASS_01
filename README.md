# Multilayer Perceptron Comprehensive Experiment Suite

This project implements a complete multilayer perceptron (MLP) training system with comprehensive experiments on XOR and binary adder datasets, as specified in the computer vision course homework.

## Files Structure

```
├── main.cpp              # Main experimental suite
├── headers/
│   ├── Complex.h         # Template complex number class
│   ├── Matrix.h          # Template matrix operations with randomization
│   └── MLP.h             # Complete MLP with training and evaluation
├── datasets/
│   ├── xor_dataset.csv           # XOR truth table (4 samples)
│   └── binary_adder_dataset.csv  # 2-bit binary adder (32 samples)
├── run                   # Bash script to compile and run experiments
├── run.ps1              # PowerShell version for Windows
├── Makefile             # Build configuration
└── README.md            # This documentation
```

## Mathematical Implementation

### Core Neural Network Components

1. **Fully-Connected Linear Layer**: `linear(x;W, b) = Wx + b`
   - Transforms input vector `x` using weight matrix `W` and bias vector `b`
   - Foundation of neural network computation

2. **Sigmoid Activation**: `σ(x) = 1/(1 + e^(-x))`
   - Non-linear function that squashes values between 0 and 1
   - Enables learning of complex, non-linear patterns

3. **Single Layer Perceptron**: `slp(x;W, b) = sigmoid(linear(x;W, b))`
   - Combines linear transformation with non-linear activation
   - Basic building block of multilayer networks

4. **Multilayer Perceptron**: `mlp(x;W₁,...,Wₙ, b₁,...,bₙ) = slp(...slp(x;W₁, b₁)...;Wₙ, bₙ)`
   - Stacks multiple perceptrons to learn complex functions
   - Each layer transforms the previous layer's output

### Training Algorithm (How the Network Learns)

1. **L2 Loss Function**: `loss(u, v) = (u - v) · (u - v)`
   - Measures how far the network's prediction `u` is from the target `v`
   - Squaring ensures positive values and penalizes large errors more

2. **Gradient Descent**: `θᵢ₊₁ = θᵢ - η∇f(θᵢ)`
   - Updates parameters `θ` (weights and biases) to minimize loss
   - `η` is learning rate (how big steps to take)
   - `∇f(θᵢ)` is the gradient (direction of steepest increase)

3. **Backpropagation**: Complete reverse-mode automatic differentiation
   - Calculates gradients by working backwards through the network
   - Uses chain rule to compute how each parameter affects the final loss

4. **Batch Processing**: Accumulates gradients over all training samples
   - Averages gradients from all samples for stable learning
   - Updates parameters once per epoch (full pass through data)

## Experimental Design

### Datasets

#### XOR Dataset (xor_dataset.csv)
- **Inputs**: 2 binary values (x1, x2)
- **Output**: 1 binary value (XOR result)
- **Size**: 4 complete samples
- **Challenge**: Non-linearly separable, requires hidden layers

#### Binary Adder Dataset (binary_adder_dataset.csv) 
- **Inputs**: 5 binary values (a0, b0, c0, a1, b1)
  - a0, a1: First 2-bit number
  - b0, b1: Second 2-bit number  
  - c0: Carry input
- **Outputs**: 3 binary values (s0, s1, c2)
  - s0, s1: Sum bits
  - c2: Carry output
- **Size**: 32 complete samples (2⁵ combinations)
- **Challenge**: Complex multi-output function

### Hyperparameter Exploration

#### XOR Experiments (10 configurations × 3 splits = 30 experiments)
- **Architectures**: 2-4-1, 2-8-1, 2-16-1, 2-4-4-1, 2-8-4-1
- **Learning Rates**: 0.1, 0.3, 0.5, 0.7
- **Epochs**: 500, 1000, 1500, 2000
- **Splits**: 50/50, 70/30, 80/20

#### Binary Adder Experiments (10 configurations × 3 splits = 30 experiments)
- **Architectures**: 5-8-3, 5-16-3, 5-32-3, 5-10-8-3, 5-16-8-3, 5-20-10-3
- **Learning Rates**: 0.1, 0.15, 0.2, 0.3, 0.5
- **Epochs**: 500, 1000, 1500, 2000
- **Splits**: 50/50, 70/30, 80/20

### Evaluation Metrics

1. **Training Loss**: L2 loss on training set
2. **Test Loss**: L2 loss on test set (generalization)
3. **Training Accuracy**: Binary classification accuracy on training set
4. **Test Accuracy**: Binary classification accuracy on test set
5. **Split Analysis**: Performance across different train/test ratios

## How to Run

### For Linux/Unix/WSL (Bash)
```bash
# Make the script executable (first time only)
chmod +x run

# Run the complete experiment suite
./run
```

### For Windows (PowerShell)
```powershell
# Run the PowerShell script
.\run.ps1

# Alternative: Run directly in PowerShell
powershell -ExecutionPolicy Bypass -File "run.ps1"
```

### For Windows (Git Bash)
```bash
# If you have Git Bash installed
bash run
```

### Manual Compilation (Any Platform)
```bash
# Linux/Unix/Git Bash
g++ -std=c++17 -Wall -Wextra -O2 -I. main.cpp -o mlp_experiments
./mlp_experiments

# Windows (MinGW/MSYS2)
g++ -std=c++17 -Wall -Wextra -O2 -I. main.cpp -o mlp_experiments.exe
mlp_experiments.exe
```

### Platform-Specific Notes
- **Linux/Unix**: Use `./run` bash script
- **Windows**: Use `.\run.ps1` PowerShell script  
- **WSL**: Use either bash or PowerShell script
- **Git Bash on Windows**: Use `bash run` command

## Program Execution Flow

### **MODIFIED VERSION**: Interactive Configuration Selection
The program now allows you to choose specific configurations instead of running all experiments:

1. **Dataset Loading**: Loads and validates XOR and Binary Adder datasets
2. **XOR Configuration Menu**: Choose from 10 different network architectures and training settings
3. **Split Ratio Selection**: Choose train/test split (50/50, 70/30, or 80/20)
4. **Binary Adder Configuration Menu**: Choose configuration for the second dataset
5. **Training Execution**: Runs only your selected experiments (much faster!)
6. **Results Analysis**: Shows detailed performance metrics for your choices

### What You'll See When Running

1. **Dataset Loading**: Confirmation of dataset sizes and dimensions
   ```
   ✓ XOR Dataset: 4 samples, 2 inputs, 1 outputs
   ✓ Binary Adder Dataset: 32 samples, 5 inputs, 3 outputs
   ```

2. **Interactive Configuration Selection**: Choose your experiment settings
   ```
   Available XOR Configurations:
   1. Small Hidden (2-4-1, LR=0.5, Epochs=1000)
   2. Medium Hidden (2-8-1, LR=0.5, Epochs=1000)
   ...
   Enter your choice (1-10): 
   ```

3. **Training Progress**: Real-time training status
   ```
   Running XOR experiment with:
   Config: Medium Hidden
   Split: 0.7
   ```

4. **Detailed Results Table**: Complete performance metrics
5. **Best Configuration Analysis**: Summary of your selected experiments
6. **Final Summary**: Overall experimental insights

### Sample Output Format
```
==========================================
EXPERIMENT RESULTS FOR XOR DATASET
==========================================
Architecture  LR     Epochs  Split  Train Loss  Test Loss   Train Acc  Test Acc   Description
-------------------------------------------------------------------------------------------------
2-4-1        0.500   1000    0.50   0.0234      0.0287      1.000      1.000      Small Hidden
2-8-1        0.500   1000    0.50   0.0156      0.0198      1.000      1.000      Medium Hidden
...
```

## Key Implementation Features

### Template-Based Design (Code Flexibility)
- **Type Safety**: Supports `double`, `float`, and `ComplexNumber<T>`
  - Same code works with different number types
  - Compiler catches type mismatches at compile time
- **Flexibility**: Easy to extend for new numeric types
  - Want to use higher precision? Just change the template parameter
- **Performance**: Compile-time optimization
  - Templates generate optimized code for each specific type

### Robust Training System (How Learning Works)
- **Complete Backpropagation**: Full gradient computation with chain rule
  - Mathematically correct implementation of gradient descent
  - Calculates exact derivatives for optimal learning
- **Validation Monitoring**: Track both training and test performance
  - Shows if model is learning (decreasing training loss)
  - Shows if model generalizes (similar test loss)
- **Hyperparameter Control**: Configurable architectures, learning rates, epochs
  - User can experiment with different network designs
  - Find optimal settings for each problem
- **Random Initialization**: Proper weight and bias initialization
  - Breaks symmetry so neurons learn different features
  - Uses appropriate random ranges for stable learning

### Comprehensive Evaluation (Understanding Results)
- **Multiple Metrics**: Loss and accuracy measurement
  - Loss: How wrong are the predictions? (continuous measure)
  - Accuracy: What percentage of predictions are correct? (discrete measure)
- **Split Validation**: Different train/test ratios for generalization analysis
  - Tests how well the model works on unseen data
  - Different splits show robustness of the learned model
- **Statistical Analysis**: Best configuration identification
  - Automatically finds which settings worked best
  - Saves time by highlighting successful approaches
- **Human-Readable Output**: Clear, formatted result presentation
  - Tables and summaries make results easy to understand
  - No need to dig through raw numbers

## What This Demonstrates (Scientific Validation)

The implementation teaches several key machine learning concepts:

1. **XOR Problem - Why Hidden Layers Matter**
   - XOR cannot be solved by a straight line (linear separator)
   - Requires hidden layers to create non-linear decision boundaries
   - Classic example showing limitations of simple perceptrons

2. **Scaling Complexity - Network Size vs Problem Difficulty**
   - XOR (2→1): Simple problem, small network works fine
   - Binary Adder (5→3): Complex problem, needs larger networks
   - Demonstrates how architecture must match problem complexity

3. **Hyperparameter Sensitivity - Why Settings Matter**
   - Learning rate too high: Network oscillates, doesn't converge
   - Learning rate too low: Network learns very slowly
   - Architecture too small: Cannot learn complex patterns
   - Architecture too large: May overfit to training data

4. **Generalization - Does It Work on New Data?**
   - Train/test splits validate model's ability to generalize
   - Good model: Similar performance on train and test sets
   - Overfitted model: Great on training, poor on test data

5. **Gradient Descent - The Learning Algorithm**
   - Complete implementation of how neural networks actually learn
   - Shows backpropagation calculating gradients automatically
   - Demonstrates convergence through iterative parameter updates

## Assignment Compliance

✅ **Controllable Parameters**: Number of layers, hidden units, learning rate, iterations
✅ **Two Datasets**: XOR (2→1) and Binary Adder (5→3) 
✅ **Multiple Splits**: 50/50, 70/30, 80/20 train/test ratios
✅ **Hyperparameter Exploration**: Comprehensive architecture and learning rate testing
✅ **Human-Readable Output**: Clear tables and analysis
✅ **Working Demonstration**: Convincing evidence of correct implementation
✅ **Automatic Differentiation**: Reverse-mode AD via backpropagation
