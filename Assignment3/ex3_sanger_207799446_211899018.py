import numpy as np
import matplotlib.pyplot as plt

def sanger():
    X = np.load('X.npy')
    M = np.load('M.npy')
    M_reshaped = M.T.reshape(2, 2, 3)

    eta = 0.0001
    epochs = 50
    num_samples = X.shape[1]

    # Starting with small random vectors
    np.random.seed(42)
    W = np.random.randn(2, 3) * 0.1

    # Learning loop (sanger rule) 
    for _ in range(epochs):
        indices = np.random.permutation(num_samples)

        for idx in indices:
            x = X[:, idx]
            
            y = W @ x
            W[0] += eta * y[0] * (x - y[0] * W[0])
            W[1] += eta * y[1] * (x - y[0] * W[0] - y[1] * W[1])

    # Calculate the final length of the vectors and the angle between them
    len_w1 = np.linalg.norm(W[0])
    len_w2 = np.linalg.norm(W[1])
    dot_product = np.dot(W[0], W[1])
    cos_theta = dot_product / (len_w1 * len_w2)
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # Plotting, with 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[0, :], X[1, :], X[2, :], color='orange', s=2, alpha=0.3, label='Training Data')

    mx, my, mz = M_reshaped[:,:,0], M_reshaped[:,:,1], M_reshaped[:,:,2]
    ax.plot_surface(mx, my, mz, color='gray', alpha=0.4)
    center = np.mean(M, axis=1)
    scale = 10

    # Draw the two calculated direction arrows (scaled up so we can see them)
    ax.quiver(center[0], center[1], center[2], W[0,0], W[0,1], W[0,2], 
            color='blue', length=scale, normalize=False, linewidth=2, label='w1')
    ax.quiver(center[0], center[1], center[2], W[1,0], W[1,1], W[1,2], 
            color='navy', length=scale, normalize=False, linewidth=2, label='w2')

    ax.text(center[0]+W[0,0]*scale, center[1]+W[0,1]*scale, center[2]+W[0,2]*scale, r'$\vec{w}^{(1)}$', color='blue')
    ax.text(center[0]+W[1,0]*scale, center[1]+W[1,1]*scale, center[2]+W[1,2]*scale, r'$\vec{w}^{(2)}$', color='navy')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Show final results in the title
    title_text = rf'$||\vec{{w}}^{(1)}||={len_w1:.4f}, ||\vec{{w}}^{(2)}||={len_w2:.4f}, \angle(\vec{{w}}^{(1)},\vec{{w}}^{(2)})={angle_deg:.2f}^\circ$'
    ax.set_title(title_text, fontsize=12)
    plt.show()

if __name__ == "__main__":
    sanger()