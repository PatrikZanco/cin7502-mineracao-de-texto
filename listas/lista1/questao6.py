import networkx as nx
import matplotlib.pyplot as plt

# cria o grafo
G = nx.DiGraph()

# nodes de entrada
inputs = ['x1', 'x2', 'x3']
weights = ['w1', 'w2', 'w3']
bias = 'b'
sum_node = 'Σ (x·w) + b'
activation = 'f(z)'
output = 'y'

G.add_nodes_from(inputs + [bias, sum_node, activation, output])

for i, w in zip(inputs, weights):
    G.add_edge(i, sum_node, label=w)

G.add_edge(bias, sum_node, label='1×b')

G.add_edge(sum_node, activation, label='z')
G.add_edge(activation, output, label='y')

pos = {
    'x1': (-2, 2),
    'x2': (-2, 1),
    'x3': (-2, 0),
    'b': (-2, -1),
    sum_node: (0, 1),
    activation: (2, 1),
    output: (4, 1)
}

plt.figure(figsize=(8, 5))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)

plt.title("Arquitetura de um Perceptron Simples")
plt.axis('off')
plt.show()
