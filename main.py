import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Compute the sigmoid function of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    return 1 / (1 + np.exp(-z))

def initialisation(X):
    """
    Initialize the weights W and bias b.

    Args:
        X (ndarray): The training data.

    Returns:
        tuple: A tuple containing the weights W and bias b.
    """
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W, b

def modele(X, W, b):
    """
    Compute the output of the model.

    Args:
        X (ndarray): The input data.
        W (ndarray): The weights.
        b (float): The bias.

    Returns:
        ndarray: The output of the model.
    """
    Z = X.dot(W) + b
    A = sigmoid(Z)
    return A

def log_loss(y, A):
    """
    Compute the logarithmic loss.

    Args:
        y (ndarray): The actual labels.
        A (ndarray): The model predictions.

    Returns:
        float: Normalized logarithmic loss.
    """
    return 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def gradients(X, A, y):
    """
    Compute gradients for optimization.

    Args:
        X (ndarray): The input data.
        A (ndarray): The model predictions.
        y (ndarray): The actual labels.

    Returns:
        tuple: A tuple containing gradients of weights (dW) and bias (db).
    """
    dW = 1/len(y) * np.dot(X.T, A - y)
    db = 1/len(y) * np.sum(A - y)
    return dW, db

def optimisation(X, W, b, A, y, learning_rate):
    """
    Optimize the weights and bias of the model.

    Args:
        X (ndarray): The input data.
        W (ndarray): The weights.
        b (float): The bias.
        A (ndarray): The model predictions.
        y (ndarray): The actual labels.
        learning_rate (float): Learning rate.

    Returns:
        tuple: A tuple containing the new weights (W) and the new bias (b).
    """
    dW, db = gradients(X, A, y)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def predict(X, W, b):
    """
    Make binary predictions.

    Args:
        X (ndarray): The input data.
        W (ndarray): The weights.
        b (float): The bias.

    Returns:
        ndarray: Array of binary predictions (True/False).
    """
    A = modele(X, W, b)
    return A >= 0.5

def regression_logistique(X, y, learning_rate=0.1, n_iter=10000):
    """
    Train a logistic regression model.

    Args:
        X (ndarray): The training data.
        y (ndarray): The actual labels.
        learning_rate (float): Learning rate.
        n_iter (int): Number of training iterations.

    Returns:
        tuple: A tuple containing the final weights (W) and the final bias (b).
    """
    # Initialization
    W, b = initialisation(X)
    loss_history = []

    # Training
    for i in range(n_iter):
        A = modele(X, W, b)
        loss_history.append(log_loss(y, A))
        W, b = optimisation(X, W, b, A, y, learning_rate)

    # Plotting the evolution of loss
    plt.plot(loss_history)
    plt.xlabel('n_iteration')
    plt.ylabel('Log_loss')
    plt.title('Evolution of errors')
    plt.show()

    return W, b
