from NeuralNetwork import NNetwork
import numpy as np

def json_to_nn(data):
    try:
        # Extract layer sizes and activation functions from JSON data
        layer_sizes = [len(layer['nodes']) for layer in data['layers']]
        activation_functions = [layer['activation'] for layer in data['layers'][:-1]]  # Exclude the output layer

        # Create an instance of NNetwork with the layer sizes and activation functions
        nn = NNetwork(len(layer_sizes), layer_sizes, activation_functions, verbose=True)

        # Now set the weights and biases from the JSON data
        for i, layer in enumerate(data['layers']):
            for j, node in enumerate(layer['nodes']):
                nn.layers[i][j].weights = np.array(node['weights'])  # Use the weights from JSON data
            nn.layers[i].bias = node['bias']  # Set the bias from JSON data

        return nn
    except Exception as e:
        print(f"Error in json_to_nn: {str(e)}")
        return None


def nn_to_json(nn):
    try:
        # Convert the NNetwork object to JSON format
        graph_data = {
            'layers': []
        }

        for i, layer in enumerate(nn.layers):
            layer_data = {
                'nodes': [],
                'activation': nn.activation_functions[i] if i < len(nn.activation_functions) else 'linear',  # Use 'linear' for output layer by default
                'bias': 0  # Bias is not used in this implementation, but can be added if needed
            }

            for node in layer:
                node_data = {
                    'weights': node.weights.tolist(),
                    'bias': node.bias
                }
                layer_data['nodes'].append(node_data)

            graph_data['layers'].append(layer_data)

        return graph_data
    except Exception as e:
        print(f"Error in nn_to_json: {str(e)}")
        return None
