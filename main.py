import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from NeuralNetwork import NNetwork 

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, learning_rate=0.01, epochs=10, verbose=1):
    """
    Melatih model dengan parameter yang diberikan.

    :param model: Objek dari NNetwork.
    :param X_train: Data training (numpy array, shape: (num_samples, num_features))
    :param y_train: Label training (numpy array, shape: (num_samples, num_classes))
    :param X_val: Data validasi (numpy array, shape: (num_samples, num_features))
    :param y_val: Label validasi (numpy array, shape: (num_samples, num_classes))
    :param batch_size: Jumlah sampel per batch saat training.
    :param learning_rate: Learning rate untuk gradient descent.
    :param epochs: Jumlah epoch untuk training.
    :param verbose: 0 = tanpa output, 1 = progress bar + training & validation loss.
    :return: Dictionary berisi histori training loss & validation loss tiap epoch.
    """
    history = {"train_loss": [], "val_loss": []}
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = num_samples // batch_size

        batch_iterator = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", disable=(verbose == 0))

        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            X_batch, y_batch = X_train[start_idx:end_idx], y_train[start_idx:end_idx]

            loss = model.backward_propagation(X_batch, y_batch, learning_rate)
            epoch_loss += loss

            if verbose == 1:
                batch_iterator.set_postfix(train_loss=loss)

        train_loss = epoch_loss / num_batches
        history["train_loss"].append(train_loss)

        val_preds = model.forward_propagation(X_val)
        val_loss = np.mean((val_preds - y_val) ** 2)
        history["val_loss"].append(val_loss)

        if verbose == 1:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

    return history


def plot_training_history(history):
    """Menampilkan grafik training loss dan validation loss per epoch."""
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # Contoh data dummy (dataset XOR)
    X_train = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    y_train = np.array([
        [0], [1], [1], [0]
    ])

    X_val = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    y_val = np.array([
        [0], [1], [1], [0]
    ])

    # Inisialisasi model
    model = NNetwork(num_of_layers=3, layer_sizes=[2, 4, 1], activation_functions=["relu", "sigmoid"], verbose=True)
    model.initialize_weights(method="xavier", seed=42)

    # Training model
    history = train_model(model, X_train, y_train, X_val, y_val, batch_size=2, learning_rate=0.1, epochs=100, verbose=1)

    # Tampilkan grafik training loss
    plot_training_history(history)
