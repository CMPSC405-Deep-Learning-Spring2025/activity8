import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Load Fashion-MNIST dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Explore activation functions (ReLU, Sigmoid, Tanh)

# First, implement the activation functions
def relu(x):
    #TODO

def sigmoid(x):
    #TODO

def tanh(x):
    #TODO

# Plot activation functions
x = np.linspace(-5, 5, 100)
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.legend()
plt.title("Activation Functions")
plt.show()

# Initialize weights and biases manually for a simple one-layer model
weights = np.random.randn(28*28, 10) * 0.01
biases = np.zeros((10,))

def cross_entropy_loss(predictions, targets):
    # predictions shape: (batch_size, 10)
    # targets shape: (batch_size,)
    m = targets.shape[0]
    # stability trick
    exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(m), targets])
    return np.sum(correct_logprobs) / m, probs

learning_rate = 0.01
epochs = 1

# Compare how different activation functions affect training (use relu, sigmoid, or tanh)
for epoch in range(epochs):
    for images, labels in trainloader:
        # Convert to NumPy
        X = images.view(images.size(0), -1).numpy()
        Y = labels.numpy()

        # Forward pass
        z = X @ weights + biases
        a = relu(z)  # TODO: Swap relu(z) with sigmoid(z) or tanh(z) to compare

        # Compute loss
        loss, probs = cross_entropy_loss(a, Y)
    
        # Backward pass (gradient wrt a)
        batch_size = X.shape[0]
        grad_output = probs
        grad_output[range(batch_size), Y] -= 1
        grad_output /= batch_size

        # Derivative of ReLU
        grad_output[z <= 0] = 0

        # Gradients for weights and biases
        grad_w = X.T @ grad_output
        grad_b = np.sum(grad_output, axis=0)

        # Update weights and biases
        weights -= learning_rate * grad_w
        biases  -= learning_rate * grad_b
