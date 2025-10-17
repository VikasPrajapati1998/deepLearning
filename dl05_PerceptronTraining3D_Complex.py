import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# ------------------- Step 0: Define perceptron -------------------
def step(x):
    return 1 if x >= 0 else 0

def Perceptron_Animate(X, y, epochs, lr=0.05):
    """Train Perceptron and save weights for animation"""
    X_aug = np.insert(X, 0, 1, axis=1)  # add bias term
    weights = np.random.randn(X_aug.shape[1]) * 0.5  # random small initialization
    weight_history = []

    for _ in tqdm(range(epochs), desc="Training Perceptron"):
        j = np.random.randint(0, X_aug.shape[0])
        y_ = step(np.dot(X_aug[j], weights))
        weights = weights + lr*(y[j] - y_)*X_aug[j]
        # Save weights every 10 epochs for smoother animation
        if _ % 10 == 0:
            weight_history.append(weights.copy())
    return weight_history

# ------------------- Step 1: Create a more complex 3D dataset -------------------
# Overlapping clusters
X, y = make_blobs(n_samples=300, centers=2, n_features=3, cluster_std=3.5, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------- Step 2: Train perceptron -------------------
n_epochs = 1000
weight_history = Perceptron_Animate(X, y, n_epochs, lr=0.05)

# ------------------- Step 3: Setup 3D animation -------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Scatter points
scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='bwr', edgecolor='k', s=50)

xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 30),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 30))

plane_surface = [None]  # to keep reference to the plane

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.set_title("3D Perceptron Decision Plane Animation")

def update(frame):
    # Remove previous surface
    if plane_surface[0] is not None:
        plane_surface[0].remove()

    w = weight_history[frame]
    bias = w[0]
    weights = w[1:]
    zz = (-bias - weights[0]*xx - weights[1]*yy)/weights[2]
    plane_surface[0] = ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    ax.set_title(f"Epoch {frame*10}")
    return plane_surface[0],

ani = FuncAnimation(fig, update, frames=len(weight_history), interval=100, blit=False)

# ------------------- Step 4: Save animation and close -------------------
ani.save("dl05_perceptron_training_3D_noisy.gif", writer=PillowWriter(fps=15))
plt.close(fig)

# ------------------- Step 5: Show final decision plane -------------------
final_w = weight_history[-1]
bias = final_w[0]
weights = final_w[1:]
zz_final = (-bias - weights[0]*xx - weights[1]*yy)/weights[2]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='bwr', edgecolor='k', s=50)
ax.plot_surface(xx, yy, zz_final, alpha=0.3, color='gray')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.set_title("Final 3D Decision Plane after Training")
plt.show()

print("Training completed.")
print("Bias:", bias)
print("Weights:", weights)
print("Animation saved.")
