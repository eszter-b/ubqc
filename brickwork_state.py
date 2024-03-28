import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


    
def brickwork_graph(n, m):
    if (m % 8 == 5):

        G = nx.DiGraph()
        matrix = np.empty([n,m], dtype=int)
        cntr = 1

        for i in range(n):
            for j in range(m):
                matrix[i][j]=cntr
                cntr += 1

        for i in range(n):
            for j in range(m-1):
                k = matrix[i][j]
                G.add_node(k, subset=i, pos=(j,n-i))
                G.add_node(k+1, subset=i, pos=(j+1,n-i))
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


def test_edges(G: nx.Graph, m):
    for edge, node in zip(G.edges, G.nodes):
        i, j = edge
        if i % m == 1:
            print(f"({i}, {j}): {edge}  node: {node}")



if __name__ == "__main__":

    n = 3
    m = 1*8+5

    G = nx.Graph()
    G = brickwork_graph(n, m)
    #test_edges(G, m)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels = True)

    plt.savefig("graph.png")