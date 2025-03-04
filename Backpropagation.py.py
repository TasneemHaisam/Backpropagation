
import numpy as np

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of the Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)
# Input layer values
i1, i2 = 0.05, 0.1  # Given input values
w1, w2, w3, w4 = 0.15, 0.2, 0.25, 0.3
w5, w6, w7, w8 = 0.4, 0.45, 0.5, 0.55
b1, b2 = 0.35, 0.6  # Bias values
target_o1, target_o2 = 0.01, 0.99

net_h1 = i1 * w1 + i2 * w2 + b1
net_h2 = i1 * w3 + i2 * w4 + b1

out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

# Output layer calculations
net_o1 = out_h1 * w5 + out_h2 * w6 + b2
net_o2 = out_h1 * w7 + out_h2 * w8 + b2

out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)

# Total error calculation
E_o1 = 0.5 * (target_o1 - out_o1) ** 2
E_o2 = 0.5 * (target_o2 - out_o2) ** 2
E_total = E_o1 + E_o2

# 2️ Backpropagation - Updating weights
learning_rate = 0.5  # Learning rate

# Compute gradient for output layer weights (w5, w6, w7, w8)
delta_o1 = (out_o1 - target_o1) * sigmoid_derivative(out_o1)
delta_o2 = (out_o2 - target_o2) * sigmoid_derivative(out_o2)

w5 -= learning_rate * delta_o1 * out_h1
w6 -= learning_rate * delta_o1 * out_h2
w7 -= learning_rate * delta_o2 * out_h1
w8 -= learning_rate * delta_o2 * out_h2

# Compute gradient for hidden layer weights (w1, w2, w3, w4)
delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(out_h1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(out_h2)

w1 -= learning_rate * delta_h1 * i1
w2 -= learning_rate * delta_h1 * i2
w3 -= learning_rate * delta_h2 * i1
w4 -= learning_rate * delta_h2 * i2

# 3️ Display results
print("\nTraining results after one update round:")
print(f"Total error before update: {E_total:.6f}")
print("\nUpdated weights:")
print(f"   w1: {w1:.4f}, w2: {w2:.4f}, w3: {w3:.4f}, w4: {w4:.4f}")
print(f"   w5: {w5:.4f}, w6: {w6:.4f}, w7: {w7:.4f}, w8: {w8:.4f}")
