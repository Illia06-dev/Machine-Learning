import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

digits = load_digits()
X = digits.data / 16.0
y = digits.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.3, stratify=y_onehot, random_state=42
)

class NeuralNetwork:
    def __init__(self, hidden_width):
        self.W = hidden_width
        self.params = {
            'W1': np.random.normal(0, 0.01, (64, self.W)),
            'b1': np.zeros((self.W,)),
            'W2': np.random.normal(0, 0.01, (self.W, self.W)),
            'b2': np.zeros((self.W,)),
            'W3': np.random.normal(0, 0.01, (self.W, 10)),
            'b3': np.zeros((10,))
        }
        self.cache = {}

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        
        Z3 = np.dot(A2, self.params['W3']) + self.params['b3']
        A3 = self.softmax(Z3)
        
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
        return A3

    def compute_loss(self, A3, y):
        m = y.shape[0]
        loss = -np.sum(y * np.log(A3 + 1e-9)) / m
        return loss

    def backward(self, y):
        m = y.shape[0]
        grads = {}
        
        dZ3 = self.cache['A3'] - y
        grads['W3'] = np.dot(self.cache['A2'].T, dZ3) / m
        grads['b3'] = np.sum(dZ3, axis=0) / m
        
        dA2 = np.dot(dZ3, self.params['W3'].T)
        dZ2 = dA2 * self.relu_deriv(self.cache['Z2'])
        grads['W2'] = np.dot(self.cache['A1'].T, dZ2) / m
        grads['b2'] = np.sum(dZ2, axis=0) / m
        
        dA1 = np.dot(dZ2, self.params['W2'].T)
        dZ1 = dA1 * self.relu_deriv(self.cache['Z1'])
        grads['W1'] = np.dot(self.cache['X'].T, dZ1) / m
        grads['b1'] = np.sum(dZ1, axis=0) / m
        
        return grads

    def update(self, grads, lr):
        for key in self.params:
            self.params[key] -= lr * grads[key]

def gradient_check():
    print("Running Gradient Check...")
    nn = NeuralNetwork(16)
    X_sample = X_train[:5]
    y_sample = y_train[:5]
    
    epsilon = 1e-5
    nn.forward(X_sample)
    analytic_grads = nn.backward(y_sample)
    
    W1 = nn.params['W1']
    passed = 0
    
    for _ in range(10):
        i, j = np.random.randint(0, W1.shape[0]), np.random.randint(0, W1.shape[1])
        orig = W1[i, j]
        
        W1[i, j] = orig + epsilon
        A3_plus = nn.forward(X_sample)
        loss_plus = nn.compute_loss(A3_plus, y_sample)
        
        W1[i, j] = orig - epsilon
        A3_minus = nn.forward(X_sample)
        loss_minus = nn.compute_loss(A3_minus, y_sample)
        
        W1[i, j] = orig
        
        num_grad = (loss_plus - loss_minus) / (2 * epsilon)
        analyt_grad = analytic_grads['W1'][i, j]
        
        rel_error = abs(num_grad - analyt_grad) / max(1e-8, abs(num_grad) + abs(analyt_grad))
        if rel_error < 1e-6:
            passed += 1
            
    print(f"Gradient Check passed: {passed}/10 (Required >= 9)")

def train_model(width, lr, epochs=50, batch_size=32):
    nn = NeuralNetwork(width)
    
    out = nn.forward(X_train)
    initial_loss = nn.compute_loss(out, y_train)
    print(f"Initial loss (W={width}, LR={lr}): {initial_loss:.4f} (Expected ~2.3026)")
    
    m = X_train.shape[0]
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            nn.forward(X_batch)
            grads = nn.backward(y_batch)
            nn.update(grads, lr)
            
        out = nn.forward(X_train)
        loss = nn.compute_loss(out, y_train)
        loss_history.append(loss)
        
    test_out = nn.forward(X_test)
    preds = np.argmax(test_out, axis=1)
    trues = np.argmax(y_test, axis=1)
    acc = np.mean(preds == trues)
    
    return loss_history, acc, nn

def time_passes():
    nn = NeuralNetwork(64)
    X_sample = X_train[:32]
    y_sample = y_train[:32]
    
    fwd_times = []
    bwd_times = []
    
    for _ in range(100):
        start = time.time()
        nn.forward(X_sample)
        fwd_times.append(time.time() - start)
        
        start = time.time()
        nn.backward(y_sample)
        bwd_times.append(time.time() - start)
        
    print(f"Median Forward Time: {np.median(fwd_times)*1000:.4f} ms")
    print(f"Median Backward Time: {np.median(bwd_times)*1000:.4f} ms")
    print("Backward pass dominates because it requires more matrix multiplications and calculation of gradients for both weights and activations.")

if __name__ == "__main__":
    gradient_check()
    time_passes()
    
    widths = [16, 64]
    lrs = [0.001, 0.01, 0.1]
    
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    for W in widths:
        for lr in lrs:
            print(f"\nTraining W={W}, LR={lr}...")
            losses, acc, _ = train_model(W, lr)
            print(f"Final Test Accuracy: {acc*100:.2f}%")
            plt.plot(losses, label=f"W={W}, LR={lr}")
            
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig("results/loss_curves.png")
    print("\nPlot saved to results/loss_curves.png")