import numpy as np
from NeuralNetwork import NNetwork

# Inisialisasi Neural Network
nn = NNetwork(3, [3, 4, 2], activation_functions=["relu", "softmax"], verbose=True)
nn.initialize_weights(method="normal", mean=0, variance=0.1, seed=42)

# Data Input dan Target
inputs = np.array([[0.1, 0.5, -0.3], 
                   [0.7, -0.1, 0.2]])
targets = np.array([[1, 0], 
                    [0, 1]])

# Jalankan Backward Propagation & Update Bobot
loss = nn.backward_propagation(inputs, targets, learning_rate=0.01)
print(f"Loss setelah Backprop: {loss:.5f}")