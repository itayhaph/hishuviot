import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename='tagged_data.npz'):
    """
    Loads data from npz file. 
    If file doesn't exist, generates synthetic data for demonstration.
    """
    if os.path.exists(filename):
        data = np.load(filename)

        # Adjust keys if the specific file has different internal names.
        keys = list(data.keys())
        x = data[keys[0]] # taking first array as inputs
        y = data[keys[1]] # taking second array as labels
        print(f"Loaded {filename} successfully.")
    else:
        print(f"Warning: {filename} not found. Generating synthetic data for testing.")
        
    return x, y

def add_bias(x):
    """Appends a column of 1s to the input matrix x for the bias term."""
    bias = np.ones((x.shape[0], 1))
    return np.hstack((bias, x))

def solve_q2():
    filename = 'tagged_data.npz'
    X, Y = load_data(filename)
    
    # 1. Randomly divide into train (70%) and test (30%)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(0.7 * num_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_raw, Y_train = X[train_idx], Y[train_idx]
    X_test_raw, Y_test = X[test_idx], Y[test_idx]
    
    # Add bias term (column of 1s) to inputs
    X_train = add_bias(X_train_raw)
    X_test = add_bias(X_test_raw)
    
    # ==========================================
    # Part A: Online Learning (Linear Neuron)
    # ==========================================
    print("Running Part A: Online Learning...")
    
    # Parameters
    epochs = 100
    eta = 0.001  # Learning rate (Chosen manually as it wasn't specified in text)
    n_features = X_train.shape[1]
    
    # Initialize weights (including bias)
    w = np.random.randn(n_features) * 0.01
    
    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        # Shuffle training data at start of each epoch (good practice for SGD)
        perm = np.random.permutation(len(Y_train))
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        # Online update: update after EACH example
        for i in range(len(Y_train)):
            x_i = X_shuffled[i]
            y_i = Y_shuffled[i]
            
            # Prediction: y_hat = w * x
            y_hat = np.dot(w, x_i)
            
            # Gradient of MSE loss: (y_hat - y) * x
            gradient = (y_hat - y_i) * x_i
            
            # Update rule
            w = w - eta * gradient
        
        # Calculate errors after each epoch
        # MSE = mean((y_pred - y_true)^2)
        mse_train = np.mean((X_train @ w - Y_train) ** 2)/2
        mse_test = np.mean((X_test @ w - Y_test) ** 2)/2
        
        train_errors.append(mse_train)
        test_errors.append(mse_test)

    # Plot for Part A
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_errors, label='Train Error')
    plt.plot(range(1, epochs + 1), test_errors, label='Test Error', linestyle='--')
    plt.title('Part A: Online Learning Error vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    solve_q2()