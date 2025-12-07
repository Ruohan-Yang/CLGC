import pandas as pd
import networkx as nx
import numpy as np
import os

def get_theoretical_network(name, n, m):
    gener_path = 'data/' + name + '_' + str(n) + '_' + str(m) + '.txt'
    if os.path.exists(gener_path):
        print('load ' + name + ' network: ' + gener_path)
        data_df = pd.read_csv(gener_path, sep=' ', header=None)
    else:
        # Given target: num_nodes n, num_edges m
        if name == 'small_world':
            k = int(2 * m / n)  # Average number of connections per node
            print(k)
            p = 0.1  # Choose an appropriate p value to control the small-world properties of the small-world model.
            GenNet = nx.watts_strogatz_graph(n, k, p)
        elif name == 'scale_free':
            # Number of edges added each time a node is added m // n
            GenNet = nx.barabasi_albert_graph(n, m // n)
        elif name == 'random_graph':
            # Estimate the probability of an edge connecting any two nodes
            p = 2 * m / (n * (n - 1))
            GenNet = nx.erdos_renyi_graph(n, p)
        else:
            print('other')
            exit(0)
        edges = list(GenNet.edges)
        data_num_nodes = n
        data_df = pd.DataFrame(np.array(edges))
        data_df.to_csv(gener_path, sep=' ', index=0, header=0)
        print('generate ' + gener_path)
        print('--------------------')
        print(name, n, m)
        print('Generate network:', name)
        print('Nodes:', data_num_nodes)
        print('Edges:', data_df.shape[0])
        print('--------------------')
    return data_df, gener_path
