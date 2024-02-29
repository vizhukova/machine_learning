import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()  # create graph obj

# define a list of nodes (node IDs)
nodes = [1, 2, 3, 4, 5]

# define a list of edges
# list of tuples, each representing an edge
# tuple (id_1, id_2) means that nodes id_1 and id_2 are connected by an edge
edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (5, 5)]

# add information to the graph object
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# draw a graph and display it
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()