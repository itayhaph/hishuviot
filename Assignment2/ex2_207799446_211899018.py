import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logisticRegression(weights,X_train,y0_train,X_val,y0_val,lamb,eta,n_epochs):
    m = X_train.shape[0]
    b = 0.0
    eps = 1e-15 # Adding epsilon to avoid log(0) in CE calculation
    w = weights.copy() # Using copy of weights so we won't affect other uses of these weights

    trainLossHistory = []
    valLossHistory = []
    trainAccHistory = []
    valAccHistory = []

    for _ in range(n_epochs):
        # Using shuffled indices
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        Y_shuffled = y0_train[indices]

        z = np.dot(X_shuffled, w) + b
        y = sigmoid(z)
        ce_loss = -np.mean(Y_shuffled * np.log(y+eps) + (1 - Y_shuffled) * np.log(1 - y + eps))
        reg_term = lamb * np.sum(np.square(w))*0.5
        trainLossHistory.append(ce_loss + reg_term)

        # Calculate training accuracy:
        # For every y_i in y: we create boolean value "train_pred_i", which is true iff y_i>0.5
        # Then convert it to int (true -> 1, false -> 0)
        train_preds = (y > 0.5).astype(int) 
        trainAccHistory.append(np.mean(train_preds == Y_shuffled))

        # Same process for validation values:
        y_val = sigmoid(np.dot(X_val, w) + b)
        ce_val = -np.mean(y0_val * np.log(y_val+eps) + (1 - y0_val) * np.log(1 - y_val+eps))
        valLossHistory.append(ce_val + reg_term)

        val_preds = (y_val > 0.5).astype(int)
        valAccHistory.append(np.mean(val_preds == y0_val))

        # gradient:
        error = y-Y_shuffled
        dw = (1 / m) * np.dot(X_shuffled.T, error) + (lamb * w)
        db = (1 / m) * np.sum(error)

        w -= eta*dw
        b -= eta*db

    return trainLossHistory, valLossHistory, trainAccHistory, valAccHistory,w 

def plotHelper(x_values, results, x_label, title_suffix, opt_val):
        # Extract final loss and accuracy for train and val
        train_ce = [r[0][-1] for r in results]
        val_ce = [r[1][-1] for r in results]
        train_acc = [r[2][-1] for r in results]
        val_acc = [r[3][-1] for r in results]
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.semilogx(x_values, train_ce, 'b-o', label='Train CE')
        ax1.semilogx(x_values, val_ce, 'r-o', label='Val CE')
        ax1.axvline(opt_val, color='k', linestyle='--', label=f'Optimal {x_label}')
        ax1.set_title(f"Cross Entropy vs {title_suffix}")
        ax1.set_xlabel(f"{x_label} (Log Scale)")
        ax1.set_ylabel("Final CE Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.semilogx(x_values, train_acc, 'b-o', label='Train Acc')
        ax2.semilogx(x_values, val_acc, 'r-o', label='Val Acc')
        ax2.axvline(opt_val, color='k', linestyle='--', label=f'Optimal {x_label}')
        ax2.set_title(f"Accuracy vs {title_suffix}")
        ax2.set_xlabel(f"{x_label} (Log Scale)")
        ax2.set_ylabel("Final Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.pause(0.5)

def plotWeightsHeatmap(weights, lamb_val):
    limit = np.max(np.abs(weights))
    plt.figure(figsize=(8, 6))
    im = plt.imshow(weights.reshape(28, 28), cmap='bwr', vmin=-limit, vmax=limit)
    plt.colorbar(im)
    plt.title(f"Weights Visualization (Lambda={lamb_val})", fontsize=12, fontweight='bold', pad=15)
    plt.axis('off')
    plt.show()

def q3():
    # Section a:
    data = np.load("binary_class.npz", allow_pickle=True)
    X_train, y0_train = data["X_train"], data["y_train"]
    X_val, y0_val = data["X_validation"], data["y_validation"]
    X_test, y0_test = data["X_test"], data["y_test"]

    plt.figure(figsize=(10, 4))

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f"\n\nrow_{i}\nLabel (y0): {y0_train[i]}", fontsize=10)
        plt.axis('off')

    plt.suptitle("Sample Images from Training Set", fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout() 
    plt.pause(0.5)
    
    # Section b:
    lamb = 0
    eta = 0.1
    w = np.random.standard_normal(len(X_train[0]))

    # Executing training process for Section c:
    t_loss, v_loss, t_acc, v_acc,_ = logisticRegression(w,X_train, y0_train, X_val, y0_val, lamb, eta, 1000)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot showing both train and validation data
    ax1.plot(t_loss, color='blue', label='Train')
    ax1.plot(v_loss, color='red', label='Val')
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(t_acc, color='blue', label='Train')
    ax2.plot(v_acc, color='red', label='Val')
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.pause(0.5)

    # Section e:
    etas = np.logspace(-4, -1, 6)
    lambdas = np.logspace(-6, -2, 6) 
    best_val_acc = -1
    opt_eta, opt_lamb = 0, 0
    
    # Using grid search
    for e in etas:
        for l in lambdas:
            _, _, _, v_acc_hist,_ = logisticRegression(w,X_train, y0_train, X_val, y0_val, l, e, 1000)
            final_acc = v_acc_hist[-1]
            if final_acc > best_val_acc:
                best_val_acc = final_acc
                opt_eta = e
                opt_lamb = l
    
    eta_range = np.logspace(np.log10(opt_eta) - 2, np.log10(opt_eta) + 2, 11)
    lamb_range = np.logspace(np.log10(opt_lamb) - 2, np.log10(opt_lamb) + 2, 11)

    # Using optimal lambda and eta
    res_eta = [logisticRegression(w,X_train, y0_train, X_val, y0_val, opt_lamb, e, 1000) for e in eta_range]
    res_lamb = [logisticRegression(w,X_train, y0_train, X_val, y0_val, l, opt_eta, 1000) for l in lamb_range]
    plotHelper(eta_range, res_eta, "Eta", "Learning Rate", opt_eta)
    plotHelper(lamb_range, res_lamb, "Lambda", "Regularization", opt_lamb)

    # Section f:
    # Using optimal lambda,eta and test values to determine the test error and accuracy
    res_test = logisticRegression(w,X_train, y0_train, X_test, y0_test, opt_lamb, opt_eta, 1000)
    print("First error value: ",res_test[1][0]," Last error value:",res_test[1][-1])
    print('First Acc:',res_test[3][0]," Last Acc:",res_test[3][-1])

    # Section g:
    # Visualize weights with optimal lambda
    final_weights_opt = res_test[4]
    plotWeightsHeatmap(final_weights_opt, opt_lamb)

    # Section h:
    # Generate and visualize weights WITHOUT regularization (lambda = 0)
    res_no_reg = logisticRegression(w, X_train, y0_train, X_test, y0_test, 0, opt_eta, 1000)
    weights_no_reg = res_no_reg[4]
    plotWeightsHeatmap(weights_no_reg, 0)

if __name__ == "__main__":
    q3()