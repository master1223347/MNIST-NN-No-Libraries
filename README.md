# MNIST Neural Network (No ML Libraries)

A simple MNIST digit classifier built from scratch:
- Python handles data loading, training loop orchestration, and evaluation
- C++ implements core neural network math (dense layers, ReLU, softmax-cross-entropy, SGD)
- `pybind11` bridges Python and C++

## Model

- Input: `784` (28x28 flattened image)
- Hidden layer: `128` units + ReLU
- Output: `10` logits (digits 0-9)
- Loss: softmax + cross-entropy
- Optimizer: SGD

## How It Works

1. Load MNIST IDX files from `data/` using `neuralpy/data.py`.
2. Shuffle training data with `shuffle_data(...)` in `neuralpy/utils.py`.
3. For each sample in `neuralpy/train.py`:
- Forward pass: `Dense(784->128) -> ReLU -> Dense(128->10)`
- Compute loss: softmax + cross-entropy
- Backward pass: softmax gradient -> dense backward -> ReLU backward -> dense backward
- Update parameters with SGD (`w -= lr * grad`)
4. After training, run inference on test data in `neuralpy/main.py` and compute test accuracy.

## Hyperparameters

- `epochs = 5`
- `lr = 0.01`
- `in_dim = 784`
- `hidden_dim = 128`
- `out_dim = 10`

## Changing Hyperparameters

- Change training length and learning rate in `neuralpy/main.py` where `train(...)` is called:
  - `epochs=...`
  - `lr=...`
- Change architecture dimensions in `neuralpy/train.py` inside `train(...)`:
  - `in_dim`
  - `hidden_dim`
  - `out_dim`

Example:

```python
# neuralpy/main.py
w1, b1, w2, b2 = train(train_images, train_labels, epochs=10, lr=0.005)
```

```python
# neuralpy/train.py
hidden_dim = 256
```

Notes:
- You do not need to rebuild C++ just to change these numeric hyperparameters.
- If you changed C++ code in `neuralcpp/` or `neuralbinding/binding.cpp`, run `build.py` again.

## Project Layout

- `neuralpy/main.py` - entry point (load data, train, test)
- `neuralpy/train.py` - training loop
- `neuralpy/data.py` - MNIST IDX file loaders
- `neuralpy/utils.py` - one-hot, init, shuffle, accuracy helpers
- `neuralcpp/*.cpp` / `neuralcpp/*.h` - NN math implementation
- `neuralbinding/binding.cpp` - pybind module bindings
- `build.py` - compiles C++ extension module
- `data/` - MNIST IDX files

```text
MNIST-NN-No-Libraries/
├── README.md
├── build.py
├── data/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── neuralbinding/
│   ├── binding.cpp
│   └── neuralbinding*.so (generated)
├── neuralcpp/
│   ├── dense.h
│   ├── dense.cpp
│   ├── relu.h
│   ├── relu.cpp
│   ├── loss.h
│   ├── loss.cpp
│   ├── optimizer.h
│   ├── optimizer.cpp
│   ├── math_utils.h
│   └── math_utils.cpp
└── neuralpy/
    ├── main.py
    ├── train.py
    ├── data.py
    ├── utils.py
    └── neuralbinding*.so (copied for runtime)
```

## Requirements

- Python 3.12+ recommended
- C++17 compiler (`c++`)
- `pybind11`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pybind11
```

## Build Extension

From project root:

```bash
./venv/bin/python build.py
```

To run `main.py` from `neuralpy/`, move/copy the built module into `neuralpy/`:

```bash
cp neuralbinding/neuralbinding*.so neuralpy/
```

## Run Training + Evaluation

```bash
cd neuralpy
../venv/bin/python main.py
```

Expected output includes per-epoch loss/accuracy and final test accuracy.

## Notes

- Training is slow in this setup. A full run can take about **30 minutes to 1 hour** depending on machine performance.
- Test workflow:
1. Run `build.py`
2. Drag/copy the generated `.so` file into `neuralpy/`
3. Run `main.py`

## Results

```text
Epoch 1: loss = 0.2244, accuracy = 0.9334
Epoch 2: loss = 0.0985, accuracy = 0.9695
Epoch 3: loss = 0.0671, accuracy = 0.9794
Epoch 4: loss = 0.0496, accuracy = 0.9845
Epoch 5: loss = 0.0366, accuracy = 0.9888
Test Accuracy: 0.9767
```
