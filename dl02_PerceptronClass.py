import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm 

# ------------------- Step 0: Define your perceptron -------------------
def step(x):
    return 1 if x >= 0 else 0

def Perceptron(X, y, epoch):
    X = np.insert(X, 0, 1, axis=1)  # add bias term as column of 1s
    weights = np.ones(X.shape[1])
    lr = 0.1

    # Use tqdm for a progress bar
    for _ in tqdm(range(epoch), desc="Training Perceptron"):
        j = np.random.randint(0, X.shape[0])
        y_ = step(np.dot(X[j], weights))
        weights = weights + lr*(y[j] - y_)*X[j]

    return weights[0], weights[1:]  # bias, weights

# ------------------- Step 1: Create a 2D dataset -------------------
X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=2.0, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------- Step 2: Train perceptron -------------------
n_epochs = 1000
start_time = time.time()
bias, weights = Perceptron(X, y, n_epochs)
end_time = time.time()
training_time = end_time - start_time

print(f"\nTraining completed in {training_time:.4f} seconds")
print("Bias:", bias)
print("Weights:", weights, "\n")

# ------------------- Step 3: Calculate predictions and accuracy -------------------
X_aug = np.insert(X, 0, 1, axis=1)
weights_full = np.insert(weights, 0, bias)
y_pred = np.array([step(np.dot(xi, weights_full)) for xi in X_aug])

accuracy = np.mean(y_pred == y)
errors = np.sum(y_pred != y)

print(f"Training Accuracy: {accuracy*100:.2f}%")
print(f"Total Misclassifications: {errors} \n")

# ------------------- Step 4: Visualize decision boundary -------------------
file_name = "dl02_perceptron_decision_boundary_custom.png"

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k', s=50)

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx = np.linspace(x_min, x_max, 200)
yy = (-bias - weights[0]*xx)/weights[1]  # x2 = (-b - w1*x1)/w2

plt.plot(xx, yy, color='black', linewidth=2, label="Decision Boundary")
plt.title("Custom Perceptron Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.savefig(file_name, dpi=300)
print(f"Decision boundary plot saved as '{file_name}'")
plt.show()
