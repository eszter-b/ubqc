import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

    
def brickwork_graph(n, m):
    if (m % 8 == 5):

        G = nx.DiGraph()
        matrix = np.empty([n,m], dtype=int)
        cntr = 1

        for j in range(m):
            for i in range(n):
                matrix[i][j]=cntr
                cntr += 1

        for i in range(n):
            for j in range(m):
                k = matrix[i][j]
                G.add_node(k, pos=(j,n-i))

                if j < m-1:
                    G.add_edge(matrix[i][j], matrix[i][j+1])
                
                if (j <= m-2) and (i < n-1):
                    if (j % 8 == 3) and (i % 2 == 0):
                        G.add_edge(matrix[i][j-1], matrix[i+1][j-1])
                        G.add_edge(matrix[i][j+1], matrix[i+1][j+1])
                    if (j % 8 == 7) and (i % 2 == 1):
                        G.add_edge(matrix[i][j-1], matrix[i+1][j-1])
                        G.add_edge(matrix[i][j+1], matrix[i+1][j+1])
                
        return G
    else: 
        print("dimension of m wrong")
        return -1

def fixed_graph_2_bit():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0), n="0", w="1")
    G.add_node(1, pos=(0, 1), n="1", w="0")
    G.add_node(2, pos=(1, 1), n="2", w="0")
    G.add_node(3, pos=(1, 0), n="3", w="1")
    G.add_node(4, pos=(2, 1), n="4", w="0")
    G.add_node(5, pos=(2, 0), n="5", w="1")
    G.add_node(6, pos=(3, 1), n="6", w="0")
    G.add_node(7, pos=(3, 0), n="7", w="1")
    G.add_node(8, pos=(4, 0), n="8", w="1")
    G.add_node(9, pos=(4, 1), n="9", w="0")

    G.add_edges_from([(0,3), (1,2), (2,3), (2,4), (3,5), (4,6), (5,7), (6,9), (7,8), (8,9)])

    return G


def test_edges(G: nx.Graph, m):
    for edge, node in zip(G.edges, G.nodes):
        i, j = edge
        if i % m == 1:
            print(f"({i}, {j}): {edge}  node: {node}")
