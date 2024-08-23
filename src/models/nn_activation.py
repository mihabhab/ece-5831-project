
import numpy as np

class Activation():
    
    def relu(self, x:np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(self, x:np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def sigmoid(self, x:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x:np.ndarray) -> np.ndarray:
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def tanh(self, x:np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, x:np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)