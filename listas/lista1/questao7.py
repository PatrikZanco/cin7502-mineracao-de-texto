import numpy as np

# Funções de ativação
def step_function(z):
    """Função de ativação degrau"""
    return np.where(z >= 0, 1, 0)

def sigmoid_function(z):
    """Função de ativação sigmoide"""
    return 1 / (1 + np.exp(-z))


class Perceptron:
    def __init__(self, n_inputs, activation="step", learning_rate=0.1, epochs=10):
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation

    def activate(self, z):
        if self.activation == "step":
            return step_function(z)
        elif self.activation == "sigmoid":
            return sigmoid_function(z)
        else:
            raise ValueError("Função de ativação desconhecida!")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activate(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.activate(z)

                # Erro
                error = target - y_pred

                # Atualização dos pesos e bias
                self.weights += self.lr * error * xi
                self.bias += self.lr * error



#dados de exemplo
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# ------------------------------
# (a) Perceptron com função degrau
# ------------------------------
print("\n=== Perceptron (Função Degrau) ===")
p1 = Perceptron(n_inputs=2, activation="step", learning_rate=0.1, epochs=10)
p1.train(X, y)

print("Pesos:", p1.weights)
print("Bias:", p1.bias)
for entrada in X:
    print(f"Entrada: {entrada} -> Saída: {p1.predict(entrada)}")

# ------------------------------
# (b) Perceptron com função sigmoide
# ------------------------------
print("\n=== Perceptron (Função Sigmoide) ===")
p2 = Perceptron(n_inputs=2, activation="sigmoid", learning_rate=0.5, epochs=10000)
p2.train(X, y)

print("Pesos:", p2.weights)
print("Bias:", p2.bias)
for entrada in X:
    print(f"Entrada: {entrada} -> Saída: {p2.predict(entrada).round(3)}")
