import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_neural_network(layers, weights=None, gradients=None):
    """
    Menampilkan struktur jaringan saraf tiruan dengan bobot dan gradien.
    
    :param layers: List jumlah neuron per layer (ex: [3, 4, 2] untuk input-3, hidden-4, output-2)
    :param weights: List matriks bobot antar layer
    :param gradients: List matriks gradien bobot (opsional)
    """
    G = nx.DiGraph()  # Graf berarah
    pos = {}  # Posisi node
    
    # Buat node per layer
    y_offset = 0
    for layer_idx, num_neurons in enumerate(layers):
        for neuron_idx in range(num_neurons):
            node_name = f"L{layer_idx}_N{neuron_idx}"
            G.add_node(node_name, layer=layer_idx)
            pos[node_name] = (layer_idx, y_offset - neuron_idx)

        y_offset -= (num_neurons - 1) / 2  # Atur jarak antar layer
    
    # Tambahkan edge berdasarkan bobot
    if weights:
        for layer_idx in range(len(weights)):
            for i, neuron_in in enumerate(weights[layer_idx]):
                for j, weight in enumerate(neuron_in):
                    source = f"L{layer_idx}_N{i}"
                    target = f"L{layer_idx + 1}_N{j}"
                    G.add_edge(source, target, weight=weight)

    # Plot jaringan
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="gray")
    
    # Tambahkan label bobot dan gradien
    edge_labels = {}
    if weights:
        for layer_idx, layer_weights in enumerate(weights):
            for i, neuron_in in enumerate(layer_weights):
                for j, weight in enumerate(neuron_in):
                    source = f"L{layer_idx}_N{i}"
                    target = f"L{layer_idx + 1}_N{j}"
                    label = f"W={weight:.2f}"
                    if gradients:
                        grad = gradients[layer_idx][i][j]
                        label += f"\n∇W={grad:.2f}"
                    edge_labels[(source, target)] = label
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Struktur Jaringan Saraf dengan Bobot dan Gradien")
    plt.show()

# Contoh pemakaian
layers = [3, 4, 2]  # 3 input, 4 hidden, 2 output
weights = [
    np.random.rand(3, 4),  # Bobot antara input → hidden
    np.random.rand(4, 2)   # Bobot antara hidden → output
]
gradients = [
    np.random.rand(3, 4) * 0.1,  # Gradien antara input → hidden
    np.random.rand(4, 2) * 0.1   # Gradien antara hidden → output
]

plot_neural_network(layers, weights, gradients)