# Makefile for MLP Training Program
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I.
TARGET = mlp_train
SOURCE = main.cpp
HEADERS = headers/Complex.h headers/Matrix.h headers/MLP.h

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean up
clean:
	rm -f $(TARGET)

# Test compilation only
test: $(SOURCE) $(HEADERS)
	$(CXX) $(CXXFLAGS) -fsyntax-only $(SOURCE)

.PHONY: all run clean test
