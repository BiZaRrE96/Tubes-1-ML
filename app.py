from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from typing import List, Optional
from NeuralNetwork import NNetwork
from NN_Json_Util import json_to_nn, nn_to_json
import pickle
import os

app = Flask(__name__)
CORS(app)

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

    # Send the graph data as JSON
    return graph_data.model_dump_json() # Use .dict() to convert Pydantic model to dict


# Route to export the current graph to a file
@app.route('/api/export', methods=['POST'])
def export_graph():
    """Export the current graph to a file."""
    try:
        if graph_data is None:
            return create_response('No graph data to export', 404)
        
        with open(GRAPH_FILE, 'wb') as f:
            pickle.dump(graph_data, f)
        
        return create_response(f'Graph exported to {GRAPH_FILE} successfully.')
    except Exception as e:
        return create_response(f"Error exporting graph: {str(e)}", 400)


# Route to import the graph from a file
@app.route('/api/import', methods=['POST'])
def import_graph():
    """Import the graph from a file."""
    global graph_data
    try:
        if os.path.exists(GRAPH_FILE):
            with open(GRAPH_FILE, 'rb') as f:
                graph_data = pickle.load(f)
            
            return create_response('Graph imported successfully.')
        else:
            return create_response(f'{GRAPH_FILE} not found.', 404)
    except Exception as e:
        return create_response(f"Error importing graph: {str(e)}", 400)


if __name__ == '__main__':
    app.run(debug=True)
