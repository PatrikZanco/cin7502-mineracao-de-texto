import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def exp_relu(z, alpha=1.0):
    """ReLU exponencial (ELU simplificada)"""
    return np.where(z > 0, z, np.exp(z) - 1)


class Perceptron:
    def __init__(self, n_inputs, activation, learning_rate=0.1, epochs=10000):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation  # função de ativação

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    def train(self, X, y):
        # treinamento usando gradiente simples
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation(z)

                # erro simples
                error = target - y_pred

                # regra de atualização
                self.weights += self.lr * error * xi
                self.bias += self.lr * error


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# ------------------------------
# (a) Perceptron - Sigmóide
# ------------------------------
print("\n=== (a) Perceptron - Sigmóide ===")
p_sigmoid = Perceptron(n_inputs=2, activation=sigmoid, learning_rate=0.5, epochs=10000)
p_sigmoid.train(X, y)
for entrada in X:
    print(f"Entrada: {entrada} -> Saída: {p_sigmoid.predict(entrada).round(3)}")

# ------------------------------
# (b) Perceptron - Tangente Hiperbólica
# ------------------------------
print("\n=== (b) Perceptron - Tangente Hiperbólica ===")
p_tanh = Perceptron(n_inputs=2, activation=tanh, learning_rate=0.5, epochs=10000)
p_tanh.train(X, y)
for entrada in X:
    print(f"Entrada: {entrada} -> Saída: {p_tanh.predict(entrada).round(3)}")

# ------------------------------
# (c) Perceptron - ReLU
# ------------------------------
print("\n=== (c) Perceptron - ReLU ===")
p_relu = Perceptron(n_inputs=2, activation=relu, learning_rate=0.01, epochs=10000)
p_relu.train(X, y)
for entrada in X:
    print(f"Entrada: {entrada} -> Saída: {p_relu.predict(entrada).round(3)}")

# ------------------------------
# (d) Perceptron - ReLU Exponencial (ELU)
# ------------------------------
print("\n=== (d) Perceptron - ReLU Exponencial ===")
p_exp_relu = Perceptron(n_inputs=2, activation=exp_relu, learning_rate=0.01, epochs=10000)
p_exp_relu.train(X, y)
for entrada in X:
    print(f"Entrada: {entrada} -> Saída: {p_exp_relu.predict(entrada).round(3)}")
