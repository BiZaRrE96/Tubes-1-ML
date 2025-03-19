import numpy as np

# Fungsi Linear
def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Fungsi ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # Turunan ReLU: 1 untuk x > 0, 0 untuk x <= 0

# Fungsi Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))  # Turunan Sigmoid

# Hyperbolic Tangent (Tanh)
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2  # Turunan Tanh

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))  # Ini hanya berlaku untuk loss MSE, gunakan Jacobian untuk cross-entropy

activation_functions = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax
}

activation_derivatives = {
    "linear": linear_derivative,
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "softmax": softmax_derivative  
}