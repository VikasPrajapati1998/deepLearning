import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# ------------------- Step 0: Define Perceptron -------------------
def step(x):
    return 1 if x >= 0 else 0

def Perceptron_Animate(X, y, epochs, lr=0.05):
    """Train Perceptron and save weights for animation"""
    X_aug = np.insert(X, 0, 1, axis=1)  # Add bias
    weights = np.random.randn(X_aug.shape[1]) * 0.5
    weight_history = []

    for _ in tqdm(range(epochs), desc="Training Perceptron"):
        j = np.random.randint(0, X_aug.shape[0])
        y_ = step(np.dot(X_aug[j], weights))
        weights = weights + lr*(y[j] - y_)*X_aug[j]
        # Save every 10 epochs
        if _ % 10 == 0:
            weight_history.append(weights.copy())
    return weight_history

# ------------------- Step 1: Create a 4D dataset -------------------
X, y = make_blobs(n_samples=400, centers=2, n_features=4, cluster_std=3.5, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------- Step 2: Reduce 4D -> 3D for visualization -------------------
pca = PCA(n_components=3)
X_3D = pca.fit_transform(X)

# ------------------- Step 3: Train Perceptron -------------------
n_epochs = 2000
weight_history = Perceptron_Animate(X, y, n_epochs, lr=0.05)

# ------------------- Step 4: Setup 3D animation -------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3D[:,0], X_3D[:,1], X_3D[:,2], c=y, cmap='bwr', edgecolor='k', s=50)

xx, yy = np.meshgrid(np.linspace(X_3D[:,0].min()-1, X_3D[:,0].max()+1, 30),
                     np.linspace(X_3D[:,1].min()-1, X_3D[:,1].max()+1, 30))

plane_surface = [None]

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title("4D Perceptron Decision Plane (PCA 3D) Animation")

def update(frame):
    if plane_surface[0] is not None:
        plane_surface[0].remove()

    w = weight_history[frame]
    bias = w[0]
    weights = w[1:]

    # Project plane onto PCA 3D space (approximation)
    # Using only the first 3 PCA components for visualization
    zz = (-bias - weights[0]*xx - weights[1]*yy)/max(weights[2], 1e-5)
    plane_surface[0] = ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    ax.set_title(f"Epoch {frame*10}")
    return plane_surface[0],

ani = FuncAnimation(fig, update, frames=len(weight_history), interval=100, blit=False)

# ------------------- Step 5: Save animation -------------------
ani.save("dl06_perceptron_training_4D.gif", writer=PillowWriter(fps=15))
plt.close(fig)

# ------------------- Step 6: Show final decision plane -------------------
final_w = weight_history[-1]
bias = final_w[0]
weights = final_w[1:]
zz_final = (-bias - weights[0]*xx - weights[1]*yy)/max(weights[2], 1e-5)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3D[:,0], X_3D[:,1], X_3D[:,2], c=y, cmap='bwr', edgecolor='k', s=50)
ax.plot_surface(xx, yy, zz_final, alpha=0.3, color='gray')
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title("Final 4D Perceptron Decision Plane (PCA 3D)")
plt.show()

print("Training completed.")
print("Bias:", bias)
print("Weights:", weights)
print("Animation saved.")
