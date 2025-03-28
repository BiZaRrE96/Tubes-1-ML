from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel
from typing import List, Optional
from NeuralNetwork import NNetwork
from NN_Json_Util import json_to_nn, nn_to_json
import numpy as np
from main import train_model, plot_training_history
import pickle
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define the shared types for Graph, Layer, and Node
class Node(BaseModel):
    weights: List[float]  # List of weights for the node

class Layer(BaseModel):
    nodes: List[Node]     # List of nodes in the layer
    bias: float           # Bias for the layer
    activation: str       # Activation function for the layer

class GraphState(BaseModel):
    layers: List[Layer]   # List of layers in the graph

# Store the graph temporarily (simulating the process)
graph_data = None
GRAPH_FILE = 'graph_model.pkl'


# Helper function to create consistent responses
def create_response(message: str, status_code: int = 200, data: Optional[dict] = None):
    """Helper function to format consistent responses."""
    response = {'message': message}
    if data:
        response['data'] = data
    return jsonify(response), status_code


# Route to train and initialize the network
@app.route('/api/train', methods=['POST'])
def train_network():
    """Receive and initialize the neural network."""
    try:
        # Receive the JSON data from the frontend
        data = request.get_json()
        
        # Validate and parse the incoming data into the GraphState model
        graph = json_to_nn(data)
        if graph is None:
            return create_response('Invalid graph data received', 400)

        # Save the graph for later use (simulate storing or training)
        global graph_data
        graph_data = graph

        # Return a success message
        return create_response('Network initialized successfully')
    
    except Exception as e:
        return create_response(f"Error initializing network: {str(e)}", 400)


# Route to fetch the graph data
@app.route('/api/get_graph', methods=['GET'])
def get_graph():
    """Send the graph data back to the frontend."""
    if graph_data is None:
        nn = NNetwork(3, [3, 4, 2], verbose=True)
        nn.initialize_weights(method="normal", mean=0, variance=0.1, seed=42, verbose=True)
        
        return create_response('No graph data available', 404)

    retval = nn_to_json(graph_data)
    # Send the graph data as JSON
    return retval


@app.route('/api/export', methods=['POST'])
def export_graph():
    """Export the current graph to a file and prompt for download."""
    try:
        if graph_data is None:
            return jsonify({'error': 'No graph data to export'}), 404

        with open(GRAPH_FILE, 'wb') as f:
            pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return send_file(GRAPH_FILE, as_attachment=True, mimetype='application/octet-stream')
    except Exception as e:
        return jsonify({'error': f"Error exporting graph: {str(e)}"}), 400

@app.route('/api/import', methods=['POST'])
def import_graph():
    """Import the graph from a file."""
    global graph_data
    try:
        # Check if the file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # If a file is provided, load the graph from it
        with open(GRAPH_FILE, 'wb') as f:
            file.save(f)  # Save the uploaded file
        
        # Now load the graph data
        with open(GRAPH_FILE, 'rb') as f:
            graph_data = pickle.load(f)
        
        return jsonify({'message': 'Graph imported successfully.'}), 200
    except Exception as e:
        return jsonify({'error': f"Error importing graph: {str(e)}"}), 400

@app.route('/api/initialize_weights', methods=['POST'])
def initialize_weights():
    """Inisialisasi bobot jaringan saraf."""
    try:
        if graph_data is None:
            return create_response('No graph data available', 404)

        data = request.get_json()
        method = data.get('method', 'zero')

        graph_data.initialize_weights(method=method, seed=42)
        return create_response(f'Weights initialized successfully with method: {method}')
    except Exception as e:
        return create_response(f"Error initializing weights: {str(e)}", 400)

@app.route('/api/start_learning', methods=['POST'])
def start_learning():
    try:
        data = request.get_json()
        learning_rate = data.get('learningRate', 0.01)
        batch_size = data.get('batchSize', 4)
        epochs = data.get('epochs', 10)
        hidden_layer_count = data.get('hiddenLayerCount', 1)
        activation_functions = data.get('activationFunctions', ["relu"] * hidden_layer_count + ["softmax"])
        activation_functions_list = [activation_functions[str(i)] for i in range(len(activation_functions))]
        initializeWeightMethod = data.get('initializeWeightMethod', 'zero')
        
        print(data.get('learningRate', learning_rate))
        print(data.get('batchSize', batch_size))
        print(data.get('epochs', epochs))
        print(data.get('hiddenLayerCount', hidden_layer_count))
        print(data.get('activationFunctionsList', activation_functions_list))
    
        
        if batch_size <= 0:
            return create_response('Batch size harus lebih besar dari nol', 400)
        
        if len(activation_functions) != hidden_layer_count + 1:
            return create_response('Jumlah fungsi aktivasi tidak sesuai dengan jumlah layer', 400)

        if graph_data is None:
            return create_response('No graph data available', 404)

        # Ambil jumlah node per layer
        TEMP = [len(layer) for layer in graph_data.layers]
        print("Jumlah node per layer (TEMP):", TEMP)
        
        input_node_count = TEMP[0]
        output_node_count = TEMP[-1]
        
        num_samples = 4  # Jumlah sampel (bisa disesuaikan)
        X_train = np.random.rand(num_samples, input_node_count)  # Data input dengan shape (num_samples, input_node_count)
        y_train = np.random.randint(0, 2, size=(num_samples, output_node_count))  # Data output dengan shape (num_samples, output_node_count)

        X_val = np.random.rand(num_samples, input_node_count)  # Data validasi input
        y_val = np.random.randint(0, 2, size=(num_samples, output_node_count))
        
        if batch_size > len(X_train):
            return create_response('Batch size tidak boleh lebih besar dari jumlah data', 400)

        if X_train.shape[0] != y_train.shape[0]:
            return create_response('Jumlah sampel pada X_train dan y_train tidak sesuai', 400)

        # Inisialisasi model
        model = NNetwork(num_of_layers=hidden_layer_count + 2, layer_sizes=TEMP, activation_functions=activation_functions_list, verbose=True)
        print("sudah inisiasi")
        model.initialize_weights(method=initializeWeightMethod, seed=42)
        
        # Training model
        history = train_model(model, X_train, y_train, X_val, y_val, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, verbose=1)
        plot_training_history(history, filename="training_history.png")
        
        return create_response('Learning started successfully', data=history)
    except Exception as e:
        return create_response(f"Error starting learning: {str(e)}", 400)
    
@app.route('/api/get_training_plot', methods=['GET'])
def get_training_plot():
    """Send the training plot to the frontend."""
    try:
        return send_file("training_history.png", mimetype='image/png', as_attachment=True)
    except Exception as e:
        return create_response(f"Error sending training plot: {str(e)}", 400)

if __name__ == '__main__':
    app.run(debug=True)
