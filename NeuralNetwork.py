# Library Imports 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os

# Functions Import
import FungsiAktivasi as FA
import FungsiLoss as FL

class NNode:
    _id_counter = 0

    def __init__(self, weights: list[float] = None, bias: float = 0.0):
        self.id = NNode._id_counter
        NNode._id_counter += 1
        self.weights = weights if weights is not None else []
        self.bias = bias
        self.gradients = []

    def __repr__(self):
        return f"NNode(id={self.id}, weights={self.weights}, bias={self.bias})"

class NNetwork:
    valid_activations = {
        "linear": FA.linear,
        "relu": FA.relu,
        "sigmoid": FA.sigmoid,
        "tanh": FA.tanh,
        "softmax": FA.softmax
    }
    
    def __init__(self, num_of_layers: int, layer_sizes: list[int], activation_functions: list[str] = None, verbose=False):
        self.layers: list[list[NNode]] = [
            [NNode(weights=[np.random.randn() for _ in range(layer_sizes[i-1])] if i > 0 else [], 
                bias=np.random.randn()) 
            for _ in range(layer_sizes[i])] 
            for i in range(num_of_layers)
        ]
        self.activation_functions: list[str] = ["sigmoid"] * num_of_layers
        self.verbose = verbose
    
    # def addLayer(self, neurons: int, activation: str = None):
    #     """
    #     params:
    #     - neurons: jumlah neuron yang akan ada di layer
    #     - activation: fungsi aktivasi yang digunakan pada layer ini
    #     """
    #     self.layers.append([NNode(weights=[]) for _ in range(neurons)])
    #     self.bias.append([0.0] * neurons)

    #     # Tambahkan gradien untuk layer baru jika bukan layer pertama
    #     if len(self.layers) > 1:
    #         self.gradients.append([[0.0] * len(self.layers[-1]) for _ in range(neurons)])
    #     else:  # Untuk layer pertama, gradien adalah list kosong
    #         self.gradients.append([])

    #     # Tentukan fungsi aktivasi untuk layer ini
    #     if activation is None:
    #         self.activation_array.append(self.default_activation)
    #     else:
    #         if activation not in NNetwork.valid_activations:
    #             raise ValueError("Fungsi aktivasi tidak dikenali")
    #         self.activation_array.append(activation)
   
    # def addNode(self, layer: int, idx: int = -1, weights : list[float] = []):
    #     """
    #         params:
    #         - layer: layer tempat menambahkan node
    #         - idx: indecs tempat menambahkan node (-1 untuk menambahkan di akhir)
    #         - weights: bobot yang diinisialisasi untuk node tersebut
    #     """
    #     if idx == -1:
    #         idx = len(self.layers)
    #     self.layers[layer].insert(idx, NNode(weights=weights))
    #     self.bias[layer].append(0.0)
    #     self.gradients[layer].append([0.0] * len(weights))
    
    def initialize_weights(self, method: str = "zero", lower: float = -0.5, upper: float = 0.5, mean: float = 0.0, variance: float = 0.1, seed: int = None, verbose: bool = False):
        if seed is not None:
            np.random.seed(seed) 

        for layer_idx in range(1, len(self.layers)):
            prev_layer_size = len(self.layers[layer_idx - 1])

            for node_idx, node in enumerate(self.layers[layer_idx]):
                num_weights = prev_layer_size
            
                if method == "zero":
                    node.weights = [0.0] * num_weights
                    node.bias = 0.0
                elif method == "uniform":
                    node.weights = np.random.uniform(lower, upper, num_weights).tolist()
                    node.bias = np.random.uniform(lower, upper)
                elif method == "normal":
                    node.weights = np.random.normal(mean, np.sqrt(variance), num_weights).tolist()
                    node.bias = np.random.normal(mean, np.sqrt(variance))
                else:
                    raise ValueError(f"Metode inisialisasi '{method}' tidak dikenali. Gunakan 'zero', 'uniform', atau 'normal'.")

                if verbose:
                    print(f"Layer {layer_idx} - Node {node_idx}: weights={node.weights}, bias={node.bias}")

    def plot_network_graph(self):
        """ Menampilkan struktur jaringan dalam bentuk graf """
        G = nx.DiGraph()
        pos = {} 
        
        y_offset = 0 
        node_labels = {}

        for layer_idx, layer in enumerate(self.layers):
            x_offset = 0
            for node in layer:
                node_id = f"L{layer_idx}N{node.id}"
                G.add_node(node_id, layer=layer_idx)
                pos[node_id] = (layer_idx, -x_offset)
                node_labels[node_id] = f"N{node.id}\nB:{node.bias:.2f}"

                if layer_idx > 0:
                    prev_layer = self.layers[layer_idx - 1]
                    for prev_node in prev_layer:
                        prev_id = f"L{layer_idx-1}N{prev_node.id}"
                        weight_idx = prev_layer.index(prev_node)
                        weight = node.weights[weight_idx]
                        G.add_edge(prev_id, node_id, weight=f"{weight:.2f}")

                x_offset += 1.5

            y_offset += 1.5

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
        
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Struktur Jaringan Saraf")
        plt.show()

    def plot_weight_distribution(self, layers: list[int]):
        """Menampilkan distribusi bobot dari layer tertentu"""
        plt.figure(figsize=(10, 5))

        found_data = False 

        for layer_idx in layers:
            if layer_idx < 1 or layer_idx >= len(self.layers):
                print(f"Layer {layer_idx} tidak valid.")
                continue

            weights = [weight for node in self.layers[layer_idx] for weight in node.weights]

            if not weights: 
                print(f"Layer {layer_idx} tidak memiliki bobot.")
                continue

            found_data = True
            plt.hist(weights, bins=20, alpha=0.6, label=f"Layer {layer_idx}")

        if not found_data:
            print("Tidak ada data bobot yang dapat ditampilkan.")
            return

        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi")
        plt.legend()
        plt.title("Distribusi Bobot")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def plot_gradient_distribution(self, layers: list[int]):
        """Menampilkan distribusi gradien bobot dari layer tertentu"""
        plt.figure(figsize=(10, 5))

        found_data = False 

        for layer_idx in layers:
            if layer_idx < 1 or layer_idx >= len(self.layers):
                print(f"Layer {layer_idx} tidak valid.")
                continue

            gradients = [grad for node in self.layers[layer_idx] for grad in node.gradients]

            if not gradients:
                print(f"Layer {layer_idx} tidak memiliki gradien.")
                continue

            found_data = True
            plt.hist(gradients, bins=20, alpha=0.6, label=f"Layer {layer_idx}")

        if not found_data:
            print("Tidak ada data gradien yang dapat ditampilkan.")
            return

        plt.xlabel("Nilai Gradien")
        plt.ylabel("Frekuensi")
        plt.legend()
        plt.title("Distribusi Gradien Bobot")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def save_model(self, filename: str):
        """Menyimpan model ke file menggunakan pickle dengan error handling."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Model berhasil disimpan ke '{filename}'")
        except Exception as e:
            print(f"❌ Gagal menyimpan model: {e}")
        
    @staticmethod
    def load_model(filename: str):
        """Memuat model dari file menggunakan pickle dengan error handling."""
        if not os.path.exists(filename):
            print(f"❌ File '{filename}' tidak ditemukan.")
            return None

        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Model berhasil dimuat dari '{filename}'")
            return model
        except (pickle.UnpicklingError, EOFError):
            print(f"❌ Gagal memuat model: File '{filename}' mungkin korup atau tidak kompatibel.")
        except Exception as e:
            print(f"❌ Terjadi kesalahan saat memuat model: {e}")

        return None

    def forward_propagation(self, inputs: np.ndarray):
        """Melakukan forward propagation pada jaringan saraf."""
        if len(inputs.shape) != 2:
            raise ValueError("Input harus berbentuk (batch_size, input_size)")

        current_input = inputs 

        for layer_idx in range(1, len(self.layers)):
            layer_output = []

            activation_function = self.activation_functions[layer_idx - 1] 

            for node in self.layers[layer_idx]:
                if len(node.weights) != current_input.shape[1]:
                    raise ValueError(f"Dimensi input layer {layer_idx} tidak sesuai dengan bobot node.")

                z = np.dot(current_input, np.array(node.weights)) + node.bias

                if activation_function == "sigmoid":
                    node_output = FA.sigmoid(z)
                elif activation_function == "relu":
                    node_output = FA.relu(z)
                elif activation_function == "tanh":
                    node_output = FA.tanh(z)
                elif activation_function == "linear":
                    node_output = FA.linear(z)
                elif activation_function == "softmax":
                    layer_output.append(z) 
                    continue
                else:
                    raise ValueError(f"Fungsi aktivasi '{activation_function}' tidak dikenali.")

                layer_output.append(node_output)

            if activation_function == "softmax":
                layer_output = FA.softmax(np.array(layer_output).T) 

            current_input = np.array(layer_output).T 

        return current_input 

    def backward_propagation(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float = 0.01):
        """Melakukan backward propagation dan memperbarui bobot dengan Gradient Descent."""
        batch_size = inputs.shape[0]

        activations = [inputs]
        current_input = inputs

        for layer_idx in range(1, len(self.layers)):
            layer_output = []
            activation_function = self.activation_functions[layer_idx - 1]

            for node in self.layers[layer_idx]:
                z = np.dot(current_input, np.array(node.weights)) + node.bias
                a = FA.activation_functions[activation_function](z)
                layer_output.append(a)

            current_input = np.array(layer_output).T
            activations.append(current_input)

        errors = [None] * len(self.layers)

        output_activations = activations[-1]
        loss_derivative = output_activations - targets

        errors[-1] = loss_derivative * FA.activation_derivatives[self.activation_functions[-1]](output_activations)

        for layer_idx in range(len(self.layers) - 2, 0, -1):
            error_signal = errors[layer_idx + 1]
            activation_derivative = FA.activation_derivatives[self.activation_functions[layer_idx - 1]](activations[layer_idx])
            errors[layer_idx] = np.dot(error_signal, np.array([node.weights for node in self.layers[layer_idx + 1]])) * activation_derivative

        for layer_idx in range(1, len(self.layers)):
            prev_activation = activations[layer_idx - 1]
            error_signal = errors[layer_idx]

            for node_idx, node in enumerate(self.layers[layer_idx]):
                node.gradients = np.dot(prev_activation.T, error_signal[:, node_idx]) / batch_size
                node.bias_gradient = np.mean(error_signal[:, node_idx], axis=0)

        self.update_weights(learning_rate)

        return np.mean(loss_derivative**2) 

    def update_weights(self, learning_rate: float = 0.01):
        """Memperbarui bobot dan bias menggunakan Gradient Descent."""
        for layer_idx in range(1, len(self.layers)): 
            for node in self.layers[layer_idx]:
                # Update bobot
                node.weights -= learning_rate * np.array(node.gradients)
                # Update bias
                node.bias -= learning_rate * node.bias_gradient
