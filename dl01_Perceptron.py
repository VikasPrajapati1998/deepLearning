# sklearn_perceptron_2d_saveplot.py
# Train a Perceptron using scikit-learn on a 2D dataset with progress bar, training time, and saved plot.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from tqdm import tqdm

# ------------------- Step 0: Specify plot filename -------------------
file_name = "dl01_perceptron_decision_boundary.png"  # change this to your preferred filename

# ------------------- Step 1: Create a 2D dataset -------------------
X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=2.0, random_state=42)

# Normalize features for stable training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------- Step 2: Load Perceptron from sklearn -------------------
p = Perceptron(
    max_iter=1,       # one iteration per partial_fit call
    eta0=0.01,
    random_state=42,
    warm_start=True,  # keep training across calls to fit/partial_fit
    tol=None          # disable automatic stopping
)

# ------------------- Step 3: Train the model with progress bar -------------------
n_epochs = 1000
classes = np.unique(y_train)
start_time = time.time()

for epoch in tqdm(range(n_epochs), desc="Training Perceptron"):
    p.partial_fit(X_train, y_train, classes=classes)

end_time = time.time()
training_time = end_time - start_time

# Total training time
print(f"Total Training Time: {training_time:.4f} seconds")
print("✅ Perceptron Training Completed\n")

# ------------------- Step 4: Evaluate and Print Details -------------------
print("Weights (coefficients):", p.coef_)
print("Bias (intercept):", p.intercept_)

# Predict on train and test data
y_train_pred = p.predict(X_train)
y_test_pred = p.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Error count
train_errors = np.sum(y_train != y_train_pred)
test_errors = np.sum(y_test != y_test_pred)
print(f"\nTraining Misclassifications: {train_errors}")
print(f"Test Misclassifications: {test_errors}")

# Classification report
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
print("Confusion Matrix (Test Data):")
print(confusion_matrix(y_test, y_test_pred))

# ------------------- Step 5: Visualize and Save Decision Boundary -------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolor='k', s=50)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = p.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.title("Perceptron Decision Boundary (sklearn)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save the plot
plt.savefig(file_name, dpi=300)
print(f"\nDecision boundary plot saved as '{file_name}'")
plt.show()
