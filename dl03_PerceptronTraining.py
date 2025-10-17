import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# ------------------- Step 0: Define perceptron -------------------
def step(x):
    return 1 if x >= 0 else 0

def Perceptron_Animate(X, y, epochs, lr=0.1):
    X_aug = np.insert(X, 0, 1, axis=1)  # Add bias column
    weights = np.ones(X_aug.shape[1])
    weight_history = []

    for _ in tqdm(range(epochs), desc="Training Perceptron"):
        j = np.random.randint(0, X_aug.shape[0])
        y_ = step(np.dot(X_aug[j], weights))
        weights = weights + lr*(y[j] - y_)*X_aug[j]
        # Save weights every 50 epochs to animate
        if _ % 50 == 0:
            weight_history.append(weights.copy())
    return weight_history

# ------------------- Step 1: Create dataset -------------------
X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=2.0, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------- Step 2: Train perceptron and save weight history -------------------
n_epochs = 2000
weight_history = Perceptron_Animate(X, y, n_epochs, lr=0.1)

# ------------------- Step 3: Setup animation -------------------
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k', s=50)
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
xx = np.linspace(x_min, x_max, 200)
line, = ax.plot([], [], color='black', linewidth=2)

ax.set_xlim(x_min, x_max)
ax.set_ylim(X[:,1].min()-1, X[:,1].max()+1)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Perceptron Decision Boundary Animation")

def update(frame):
    w = weight_history[frame]
    bias = w[0]
    weights = w[1:]
    yy = (-bias - weights[0]*xx)/weights[1]
    line.set_data(xx, yy)
    ax.set_title(f"Epoch {frame*50}")
    return line,

ani = FuncAnimation(fig, update, frames=len(weight_history), interval=200, blit=True)

# ------------------- Step 4: Save animation and show final result -------------------
ani.save("dl03_perceptron_training.gif", writer=PillowWriter(fps=10))  # saves animation as GIF
plt.close(fig)  # close the animation figure automatically

# ------------------- Step 5: Show final decision boundary -------------------
# Use the last weights to show final line
final_w = weight_history[-1]
bias = final_w[0]
weights = final_w[1:]

print(f"\nTraining completed.")
print("Bias:", bias)
print("Weights:", weights, "\n")

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k', s=50)
yy_final = (-bias - weights[0]*xx)/weights[1]
plt.plot(xx, yy_final, color='black', linewidth=2, label="Final Decision Boundary")
plt.title("Final Decision Boundary after Training")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

print("Animation saved.")
