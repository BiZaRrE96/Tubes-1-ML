from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel
from typing import List, Optional
from NeuralNetwork import NNetwork
from NN_Json_Util import json_to_nn, nn_to_json
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

if __name__ == '__main__':
    app.run(debug=True)
