import numpy as np
from NeuralNetwork import NNetwork
from NN_Json_Util import nn_to_json

# Inisialisasi Neural Network
nn = NNetwork(3, [3, 4, 2], verbose=True)
nn.initialize_weights(method="normal", mean=0, variance=0.1, seed=42, verbose=True)
print("\n")

# nn.plot_network_graph()
# nn.plot_weight_distribution(layers=[1, 2])
# nn.plot_gradient_distribution(layers=[1, 2])

nn.save_model("my_model")
loaded_nn = NNetwork.load_model("my_model")

inputs = np.array([
    [0.1, 0.5, -0.3],
    [0.7, -0.1, 0.2]
])
output = nn.forward_propagation(inputs)
print(output) 

inputs = np.array([
    [0.1, 0.5, -0.3],  
    [0.7, -0.1, 0.2]    
])
targets = np.array([
    [1, 0], 
    [0, 1]  
])
loss = nn.backward_propagation(inputs, targets, learning_rate=0.01)
print(f"Loss setelah Backprop: {loss:.5f}")