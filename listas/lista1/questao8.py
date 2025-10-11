import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

input_layer = ['x1', 'x2', 'x3']
hidden_layer = ['h1', 'h2', 'h3', 'h4']
output_layer = ['y']

# Adicionar nós
G.add_nodes_from(input_layer + hidden_layer + output_layer)

for i in input_layer:
    for h in hidden_layer:
        G.add_edge(i, h)

for h in hidden_layer:
    for o in output_layer:
        G.add_edge(h, o)


pos = {
    'x1': (-3, 3),
    'x2': (-3, 2),
    'x3': (-3, 1),
    'h1': (0, 3.5),
    'h2': (0, 2.5),
    'h3': (0, 1.5),
    'h4': (0, 0.5),
    'y': (3, 2)
}

# exibir grafico
plt.figure(figsize=(8, 5))
nx.draw(G, pos, with_labels=True, node_size=2500, node_color='lightblue', arrows=True, font_size=10)
plt.title("Arquitetura de uma Rede Neural MLP (3 Camadas)")
plt.axis('off')
plt.show()
