import numpy as np

# Fungsi Linear
def linear(x):
  return x

# Fungi ReLU
def relu(x):
  return np.maximum(0, x)

# Fungsi Sigmoid
def sigmoid(x):
  return 1 + (1 + np.exp(-x))

# Hyperbolic Tangent
def tanh(x):
  return np.tanh(x)

# Softmax
def softmax(x):
  exp_x = np.exp(x - np.max(x))
  return exp_x / np.sum(exp_x, axis=0, keepdims=True)