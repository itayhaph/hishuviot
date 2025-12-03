import numpy as np
import matplotlib.pyplot as plt

def load_data(filename='tagged_data.npz'):
    data = np.load(filename)
    keys = list(data.keys())
    x = data[keys[0]] # taking first array as inputs
    y = data[keys[1]] # taking second array as labels
        
    return x, y

def add_bias(x):
    # Appending a column of 1s to the input matrix x for the bias term
    bias = np.ones((x.shape[0], 1))
    return np.hstack((bias, x))

def solve_q2_a():
    filename = 'tagged_data.npz'
    X, Y = load_data(filename)
    
    # Randomly divide the samples into train (70%) and test (30%)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(0.7 * num_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_raw, Y_train = X[train_idx], Y[train_idx]
    X_test_raw, Y_test = X[test_idx], Y[test_idx]
    
    X_train = add_bias(X_train_raw)
    X_test = add_bias(X_test_raw)
    
    epochs = 100
    eta = 0.00001
    n_features = X_train.shape[1]
    
    # Initialize weights (including bias)
    w = np.random.randn(n_features) 
    train_errors = []
    test_errors = []
    print(len(X),len(X[0]))
    
    for _ in range(epochs):
        errorSum = 0
        # Shuffle training data at start of each epoch
        perm = np.random.permutation(len(Y_train))
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        # Online update: (updating after each example)
        for i in range(len(Y_train)):
            x_i = X_shuffled[i]
            y_i = Y_shuffled[i]
            
            y_hat = np.dot(w, x_i)
            gradient = (y_hat - y_i) * x_i
            
            errorSum += ((y_hat-y_i)**2)/2
            # Update rule
            w = w - eta * gradient
        
        # Calculate errors after each epoch
        # MSE = mean((y_pred - y_true)^2)/2
        mse_train = errorSum/len(X_train)
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
    solve_q2_a()