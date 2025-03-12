import numpy as np
from math import pow, exp

class NNode :
    def __init__(self, weights : list[float] = []):
        self.id = 0
        self.weights: list[float] = weights
        self.gradients: list[float] = []

class NNetwork :
    # Probably disimpan sebagai semacam list of lists, dimana satu NNetwork terdiri dari X lists dimana masing2 list
    # Ialah suatu layer tersendiri
    # Lalu satu node direpresentasikan dengan list of weights ke arah kanan?
    valid_activations = ["linear","relu","sigmoid","tanh","softmax"]
    
    def __init__(self, default_activation : str = "sigmoid", verbose = False):
        self.layers: list[list[NNode]] = []
        self.bias: list[list[float]] = []
        self.gradients: list[list[float]] = []
        self.activation_array: list[str] = []
        self.default_activation: str = default_activation
        self.verbose: bool = verbose
    
    def activate(self,tempresult : list[float], method : str):
        if method == "sigmoid" :
            for i in range(len(tempresult)):
                tempresult[i] = (1 / (1 + exp(-tempresult[i])))
        elif method == "none":
            pass
        else :
            raise Exception("Undefined activation method")
        if (self.verbose) :
                print(tempresult)
        
    def getLayerCount(self):
        return len(self.layers)
    
    # Layer
    def addLayer(self, preferred_activation : str = None):
        self.layers.append([])
        self.bias.append([])
        self.gradients.append([])
        if (preferred_activation == None):
            self.activation_array.append(self.default_activation)
        else:
            if (not (preferred_activation in NNetwork.valid_activations)):
                raise KeyError
            self.activation_array.append(preferred_activation)
        
    def removeLayer(self,layer : int):
        try :
            self.layers.pop(layer)
            self.activation_array.pop(layer)
        except IndexError:
            print("Bruh illegal")
            
    def getLayer(self,layer : int):
        try :
            return self.layers[layer]    
        except IndexError:
            print("Bruh illegal")
            
    # Nodes
    def addNode(self, layer : int, idx : int = -1, weights : list[float] = []):
        if (idx == -1) : # Set index to last if unsupplied
            idx = len(self.layers)
        self.layers[layer].insert(idx,NNode(weights=weights))
        self.bias[layer].append(0.0)
        self.gradients[layer].append([0.0] * len(weights))
        
    def popNode(self, layer : int, idx : int):
        l = self.getLayer(layer)
        if idx >= len(l):
            raise IndexError
        
        # Iterate for every layer before it if exists
        if (layer > 0 and layer < self.getLayerCount()):
            # cleanse layer-1
            for i in range(len(self.getLayer(layer-1))):
                #For every node in layer-1
                self.layers[layer-1][i].weights.pop(idx)
        
        self.layers[layer].pop(idx)
    
    # Inisiasi Bobot dan Bias
    def initialize_weights(self, method: str = "zero", lower: float = -0.5, upper: float = 0.5, mean: float = 0.0, variance: float = 0.1, seed: int = None):
        np.random.seed(seed)
        # Inisiasi bobot
        for layer_idx in range(len(self.layers) - 1): # Skip output layer
            for node in self.layers[layer_idx]:
                num_weights = len(self.layers[layer_idx + 1]) # Weight = Jumlah neuron di layer berikutnya
                if method == "zero":
                    node.weights = [0.0] * num_weights
                elif method == "uniform":
                    node.weights = list(np.random.uniform(lower, upper, num_weights))
                elif method == "normal":
                    node.weights = list(np.random.normal(mean, np.sqrt(variance), num_weights))
                else:
                    raise ValueError("Metode inisialisasi tidak dikenali")
        
        # Inisiasi bias
        for layer_idx in range(len(self.bias) - 1):
            num_biases = len(self.layers[layer_idx + 1])
            if method == "zero":
                self.bias[layer_idx] = [0.0] * num_biases
            elif method == "uniform":
                self.bias[layer_idx] = list(np.random.uniform(lower, upper, num_biases))
            elif method == "normal":
                self.bias[layer_idx] = list(np.random.normal(mean, np.sqrt(variance), num_biases))
            else:
                raise ValueError("Metode inisialisasi tidak dikenali")
    
    # Menampilkan bobot setiap layer
    def print_weights(self):
        for i, layer in enumerate(self.layers[:-1]):
            print(f"Layer {i}:")
            for j, node in enumerate(layer):
                print(f"  Neuron {j}: Weights={node.weights}")
            print(f"  Bias: {self.bias[i]}")
        print()
    
    # Menampilkan gradient tiap layer
    def print_gradients(self, layers_to_print: list[int]):
        for layer_idx in layers_to_print:
            if layer_idx >= len(self.gradients) or layer_idx < 0:
                print(f"Layer {layer_idx} tidak valid.")
                continue
            
            print(f"Gradien Layer {layer_idx}:")
            for j, node_gradients in enumerate(self.gradients[layer_idx]):
                print(f"  Neuron {j}: Gradients={node_gradients}")
            print()