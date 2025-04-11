# Library Imports 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
from contextlib import suppress

# Functions Import
import FungsiAktivasi as FA
import FungsiLoss as FL

class NNode:
    _id_counter = 0

    def __init__(self, weights: list[float] = None, bias: float = 0.0):
        self.id = NNode._id_counter
        NNode._id_counter += 1
        self.weights = np.array(weights) if weights is not None else np.array([]) 
        self.bias = bias
        self.gradients = np.zeros_like(self.weights) 
        self.bias_gradient = 0.0 
    
    def reset_gradients(self):
        """Reset gradien bobot dan bias setelah update parameter."""
        self.gradients = np.zeros_like(self.weights)
        self.bias_gradient = 0.0

    def __repr__(self):
        return f"NNode(id={self.id}, weights={self.weights}, bias={self.bias})"

class NNetwork:
    def __init__(self, num_of_layers: int, layer_sizes: list[int], activation_functions: list[str] = None, verbose=False, weights: list[list[list[float]]] = None, biases: list[list[float]] = None):
        self.layer_sizes = layer_sizes 
        self.verbose = verbose

        if activation_functions is None:
            activation_functions = ["sigmoid"] * (num_of_layers - 2) + ["softmax"]  # Default: Sigmoid untuk hidden, Softmax untuk output
        elif len(activation_functions) != num_of_layers-1:
            raise ValueError(f"Jumlah fungsi aktivasi harus {num_of_layers-1}, bukan {len(activation_functions)}.")
        self.activation_functions = activation_functions

        # Inisialisasi layer
        self.layers: list[list[NNode]] = []
        for i in range(len(layer_sizes)):
            print("inisiasi layer", i)
            layer = [
                NNode(
                    weights=np.random.randn(layer_sizes[i-1]) if i > 0 else [],  
                    bias=np.random.randn()
                )
                for _ in range(layer_sizes[i])
            ]
            print("inisiasi node selesai")
            self.layers.append(layer)
        print("inisiasi layer selesai")
        
        # Print Struktur Jika Verbose
        if self.verbose:
            print(f"✅ Jaringan saraf dengan {num_of_layers} layer berhasil dibuat!")
            for i, layer in enumerate(self.layers):
                if i == 0:
                    print(f"🔹 Layer {i} (Input) - {len(layer)} neurons")
                else:
                    print(f"🔹 Layer {i} - {len(layer)} neurons, Aktivasi: {self.activation_functions[i-1]}")
    
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
        rng = np.random.default_rng(seed)  

        for layer_idx in range(1, len(self.layers)):
            prev_layer_size = len(self.layers[layer_idx - 1])

            for node_idx, node in enumerate(self.layers[layer_idx]):
                num_weights = prev_layer_size 

                # Mapping metode inisialisasi
                weight_init_methods = {
                    "zero": lambda: np.zeros(num_weights),
                    "uniform": lambda: rng.uniform(lower, upper, num_weights),
                    "normal": lambda: rng.normal(mean, np.sqrt(variance), num_weights),
                    "xavier": lambda: rng.normal(0, np.sqrt(1 / prev_layer_size), num_weights),  # Xavier untuk sigmoid/tanh
                    "he": lambda: rng.normal(0, np.sqrt(2 / prev_layer_size), num_weights)  # He untuk ReLU
                }

                if method not in weight_init_methods:
                    raise ValueError(f"Metode inisialisasi '{method}' tidak dikenali. Gunakan 'zero', 'uniform', 'normal', 'xavier', atau 'he'.")

                # Set bobot dan bias
                node.weights = weight_init_methods[method]()
                node.bias = rng.normal(0, np.sqrt(variance)) if method in ["normal", "xavier", "he"] else rng.uniform(lower, upper)

                if verbose:
                    print(f"Layer {layer_idx} - Node {node_idx}: weights={node.weights.tolist()}, bias={node.bias:.4f}")

    def plot_network_graph(self):
        """Menampilkan struktur jaringan dalam bentuk graf visual"""
        G = nx.DiGraph()
        pos = {}  
        node_labels = {}

        layer_spacing = 2.0  
        node_spacing = 1.5   
        node_colors = [] 

        color_map = ["lightgreen", "lightblue", "salmon"] 

        for layer_idx, layer in enumerate(self.layers):
            for node_idx, node in enumerate(layer):
                node_id = f"L{layer_idx}N{node_idx}" 

                G.add_node(node_id, layer=layer_idx)
                pos[node_id] = (layer_idx * layer_spacing, -node_idx * node_spacing)
                node_labels[node_id] = f"N{node_idx}\nB:{node.bias:.2f}"

                if layer_idx == 0:
                    node_colors.append(color_map[0]) 
                elif layer_idx == len(self.layers) - 1:
                    node_colors.append(color_map[2]) 
                else:
                    node_colors.append(color_map[1])  

                # Hubungkan ke layer sebelumnya
                if layer_idx > 0:
                    prev_layer = self.layers[layer_idx - 1]
                    for prev_idx, prev_node in enumerate(prev_layer):
                        prev_id = f"L{layer_idx-1}N{prev_idx}"
                        weight = node.weights[prev_idx]  

                        G.add_edge(prev_id, node_id, weight=f"{weight:.2f}")

        plt.figure(figsize=(12, 6))
        nx.draw(
            G, pos, with_labels=True, labels=node_labels, node_color=node_colors,
            edge_color="gray", node_size=2000, font_size=10
        )

        # Tambahkan label bobot pada edge
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Struktur Jaringan Saraf Tiruan", fontsize=14)
        plt.show()

    def plot_weight_distribution(self, layers: list[int], show_grid: bool = True):
        """Menampilkan distribusi bobot dari layer tertentu dengan statistik tambahan."""
        plt.figure(figsize=(10, 5))

        found_data = False  
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))  # Warna otomatis

        for idx, layer_idx in enumerate(layers):
            if layer_idx < 1 or layer_idx >= len(self.layers):
                print(f"Layer {layer_idx} tidak valid.")
                continue

            # Ambil semua bobot dari layer yang dipilih
            weights = np.array([weight for node in self.layers[layer_idx] for weight in node.weights])

            if weights.size == 0: 
                print(f"Layer {layer_idx} tidak memiliki bobot.")
                continue

            found_data = True

            # Hitung statistik
            mean = np.mean(weights)
            std = np.std(weights)

            plt.hist(weights, bins=20, alpha=0.6, label=f"Layer {layer_idx} (μ={mean:.3f}, σ={std:.3f})", color=colors[idx])

        if not found_data:
            print("Tidak ada data bobot yang dapat ditampilkan.")
            return

        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi")
        plt.legend()
        plt.title("Distribusi Bobot per Layer")
        if show_grid:
            plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def plot_gradient_distribution(self, layers: list[int], show_grid: bool = True):
        """Menampilkan distribusi gradien dari layer tertentu dengan statistik tambahan."""
        plt.figure(figsize=(10, 5))

        found_data = False  
        colors = plt.cm.plasma(np.linspace(0, 1, len(layers))) 

        for idx, layer_idx in enumerate(layers):
            if layer_idx < 1 or layer_idx >= len(self.layers):
                print(f"Layer {layer_idx} tidak valid.")
                continue

            gradients = np.array([grad for node in self.layers[layer_idx] for grad in node.gradients])

            if gradients.size == 0: 
                print(f"Layer {layer_idx} tidak memiliki gradien.")
                continue

            found_data = True

            mean = np.mean(gradients)
            std = np.std(gradients)

            plt.hist(gradients, bins=20, alpha=0.6, label=f"Layer {layer_idx} (μ={mean:.3f}, σ={std:.3f})", color=colors[idx])

        if not found_data:
            print("Tidak ada data gradien yang dapat ditampilkan.")
            return

        plt.xlabel("Nilai Gradien")
        plt.ylabel("Frekuensi")
        plt.legend()
        plt.title("Distribusi Gradien Bobot per Layer")
        if show_grid:
            plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def save_model(self, filename: str, verbose: bool = True):
        """Menyimpan model ke file menggunakan pickle dengan error handling."""
        if not filename.endswith(".pkl"):
            filename += ".pkl" 

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"✅ Model berhasil disimpan ke '{filename}'")
        except Exception as e:
            if verbose:
                print(f"❌ Gagal menyimpan model: {e}")
        
    @staticmethod
    def load_model(filename: str, verbose: bool = True):
        """Memuat model dari file menggunakan pickle dengan error handling."""
        if not filename.endswith(".pkl"):
            filename += ".pkl" 

        if not os.path.exists(filename):
            if verbose:
                print(f"❌ File '{filename}' tidak ditemukan.")
            return None

        with suppress(pickle.UnpicklingError, EOFError, Exception):
            with open(filename, 'rb') as f:
                model = pickle.load(f)

            if isinstance(model, NNetwork):  
                if verbose:
                    print(f"✅ Model berhasil dimuat dari '{filename}'")
                return model
            else:
                if verbose:
                    print(f"❌ File '{filename}' bukan model `NNetwork` yang valid.")

        if verbose:
            print(f"❌ Gagal memuat model: File '{filename}' mungkin korup atau tidak kompatibel.")
        return None

    def forward_propagation(self, inputs: np.ndarray):
        """Melakukan forward propagation pada jaringan saraf."""
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        if len(inputs.shape) != 2:
            raise ValueError("Input harus berbentuk (batch_size, input_size)")

        current_input = inputs

        for layer_idx in range(1, len(self.layers)):
            activation_function = self.activation_functions[layer_idx - 1]

            weights = np.array([node.weights for node in self.layers[layer_idx]])
            biases = np.array([node.bias for node in self.layers[layer_idx]])

            z = np.dot(current_input, weights.T) + biases

            if activation_function == "sigmoid":
                current_input = FA.sigmoid(z)
            elif activation_function == "relu":
                current_input = FA.relu(z)
            elif activation_function == "tanh":
                current_input = FA.tanh(z)
            elif activation_function == "linear":
                current_input = FA.linear(z)
            elif activation_function == "softmax":
                current_input = FA.softmax(z) 
            else:
                raise ValueError(f"Fungsi aktivasi '{activation_function}' tidak dikenali.")

        return current_input  

    def backward_propagation(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float = 0.01, reg_type = "L1"):
        """
        Melakukan backward propagation dan memperbarui bobot menggunaka Gradient Descent."""
        batch_size = inputs.shape[0]

        if batch_size == 0:
            raise ValueError("Batch size tidak boleh nol")

        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        if len(inputs.shape) != 2:
            raise ValueError("Input harus berbentuk (batch_size, input_size)")

        # Forward Pass
        activations = [inputs]
        current_input = inputs

        for layer_idx in range(1, len(self.layers)):
            activation_function = self.activation_functions[layer_idx - 1]
            
            weights = np.array([node.weights for node in self.layers[layer_idx]])
            biases = np.array([node.bias for node in self.layers[layer_idx]])

            z = np.dot(current_input, weights.T) + biases

            if activation_function == "sigmoid":
                current_input = FA.sigmoid(z)
            elif activation_function == "relu":
                current_input = FA.relu(z)
            elif activation_function == "tanh":
                current_input = FA.tanh(z)
            elif activation_function == "linear":
                current_input = FA.linear(z)
            elif activation_function == "softmax":
                current_input = FA.softmax(z)  
            else:
                raise ValueError(f"Fungsi aktivasi '{activation_function}' tidak dikenali.")

            activations.append(current_input)

        # Backward Pass: Hitung Gradien
        errors = [None] * len(self.layers)

        output_activations = activations[-1]
        loss_derivative = output_activations - targets  

        if self.activation_functions[-1] == "softmax":
            errors[-1] = loss_derivative  
        else:
            errors[-1] = loss_derivative * FA.activation_derivatives[self.activation_functions[-1]](output_activations)

        for layer_idx in range(len(self.layers) - 2, 0, -1): 
            error_signal = errors[layer_idx + 1]
            activation_derivative = FA.activation_derivatives[self.activation_functions[layer_idx - 1]](activations[layer_idx])
            
            weights_next_layer = np.array([node.weights for node in self.layers[layer_idx + 1]])
            errors[layer_idx] = np.dot(error_signal, weights_next_layer) * activation_derivative

        for layer_idx in range(1, len(self.layers)):
            prev_activation = activations[layer_idx - 1]
            error_signal = errors[layer_idx]

            for node_idx, node in enumerate(self.layers[layer_idx]):
                node.gradients = np.dot(prev_activation.T, error_signal[:, node_idx]) / batch_size

                node.bias_gradient = np.mean(error_signal[:, node_idx], axis=0)

        self.update_weights(learning_rate, reg_type=reg_type, reg_lambda=0.01)

        return np.mean(loss_derivative**2)  

    def update_weights(self, learning_rate: float = 0.01, reg_type: str = None, reg_lambda: float = 0.0):
        """Update weights and biases using gradient descent with optional L1 or L2 regularization, then reset gradients."""
        for layer_idx in range(1, len(self.layers)):
            for node in self.layers[layer_idx]:
                # Calculate regularization term for weights if specified.
                if reg_type == "L2":
                    reg_term = reg_lambda * node.weights
                elif reg_type == "L1":
                    reg_term = reg_lambda * np.sign(node.weights)
                else:
                    reg_term = 0.0

                # Update weights incorporating regularization penalty.
                node.weights -= learning_rate * (node.gradients + reg_term)
                node.bias -= learning_rate * node.bias_gradient

                # Reset gradients after update.
                node.reset_gradients()