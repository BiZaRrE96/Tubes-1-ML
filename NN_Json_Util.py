from NeuralNetwork import NNetwork
import numpy as np

def json_to_nn(data):
    try:
        # Extract layer sizes and activation functions from JSON data
        layer_sizes = [len(layer['nodes']) for layer in data['layers']]
        activation_functions = [layer['activation'] for layer in data['layers'][1:]]  # Exclude the output layer, assuming each layer uses its own activation function and not the previous

        # Create an instance of NNetwork with the layer sizes and activation functions
        nn = NNetwork(len(layer_sizes), layer_sizes, activation_functions, verbose=True)

        for i in range(1,len(layer_sizes)): # For every layer after 0
            biases = data['biases'][i-1]
            for j in range(layer_sizes[i]): # For every node in the layer
                for k in range(layer_sizes[i-1]): # For every node in the previous layer
                    nn.layers[i][j].weights[k] = data['layers'][i-1]['nodes'][k]['weights'][j]
                nn.layers[i][j].bias = biases['value']
                nn.layers[i][j].bias_gradient = biases['weights'][j]
        return nn
    except Exception as e:
        print(f"Error in json_to_nn: {str(e)}")
        return None


def nn_to_json(nn):
    try:
        # Convert the NNetwork object to JSON format
        graph_data = {
            'layers': [],
            'biases': []
        }

        for i, layer in enumerate(nn.layers): # For every layer
            layer_data = {
                'nodes': [],
                'activation': nn.activation_functions[i-1] if i > 0 else 'linear'  # Use 'linear' for output layer by default
            }
            
            for ix, _node in enumerate(layer): # For every node in the layer
                if (i < len(nn.layers)-1):
                    node_data = {
                        'weights': [nextnode.weights[ix] for nextnode in nn.layers[i+1] if i < len(nn.layers)-1], # For every node in the next layer, get corresponding weight
                    }
                else:
                    node_data = {
                        'weights': []
                    }
                layer_data['nodes'].append(node_data)

            graph_data['layers'].append(layer_data)

            # Add biases for all layers except the input layer
            # Asuming a layer has the same bias but different weights
            if i > 0:
                bias_data = {
                    'value': layer[0].bias,
                    'weights': [node.bias_gradient for node in layer]
                }
                graph_data['biases'].append(bias_data)

        return graph_data
    except Exception as e:
        print(f"Error in nn_to_json: {str(e)}")
        return None
