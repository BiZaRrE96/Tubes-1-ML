from NeuralNetwork import NNetwork
from sklearn.datasets import fetch_openml
import numpy as np
from tqdm import tqdm  

# pinputs = np.array([
#     [0.1, 0.5, -0.3],  
#     [0.7, -0.1, 0.2]    
# ])
# ptargets = np.array([
#     [1, 0], 
#     [0, 1]  
# ])

# nnp = NNetwork(3, [3, 2, 2], verbose=True)
# nnp.initialize_weights(method="normal", mean=0, variance=0.1, seed=42, verbose=True)
# result = nnp.backward_propagation(pinputs,ptargets,learning_rate=0.01)
# print(result)

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X= X / 255
y = np.array(y).astype(int)  # Convert to integers
num_classes = 10  # since labels are from 0 to 9
y = np.eye(num_classes)[y]

X = X[:100]
y = y[:100]

nn = NNetwork(3, [784, 10, 10], verbose=True)
nn.initialize_weights(method="normal", mean=0, variance=0.1, seed=42, verbose=True)

loss = nn.backward_propagation(X, y, learning_rate=0.01)
print(f"Loss setelah Backprop: {loss:.5f}")

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

Xtrain = X[:80]
ytrain = y[:80]
Xval = X[80]
yval = y[80]

history = train_model(nn, Xtrain, ytrain, Xval, yval, batch_size=20, learning_rate=0.01, epochs=10, verbose=1)