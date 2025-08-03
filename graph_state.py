import networkx as nx
import matplotlib.pyplot as plt


def brickwork(H: int, W: int):
    """Generate a brickwork graph

    Generate a brickwork graph with a given width and height

    :H:int, height (number of input qubits)
    :W:int, width (or number of qubit columns)
    """
    G = nx.Graph()

    # add vertices
    G.add_nodes_from(range(0,H*W))

    # add vertical edges
    for w in range(2,W,2):
        for h in range(0,H-1):
            if (2 == w % 8 or 4 == w % 8) and 0 == h % 2:
                G.add_edge(h + w*H,(h+1) + w*H)

            if (6 == w % 8 or 0 == w % 8) and 1 == h % 2:
                G.add_edge(h + w*H, (h+1) + w*H)

    # add horizontal edges
    for w in range(1,W):
        for h in range(0,H):
            G.add_edge(h + (w-1)*H,h + w*H)

    # input
    I = set(range(0,H))
    O = set(range(len(G.nodes)-H,len(G.nodes)))

    return G,I,O


def grover_4_element():
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
    I = {0, 1}
    O = {8, 9}

    return G, I, O


def deutsch_jozsa():
    """
    For balanced Deutsch-Jozsa algorithm with input angles: phi = [0.0, pi/2, pi, 0.0]
    """
    G = nx.Graph()

    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(1, 1))
    G.add_node(3, pos=(2, 1))

    G.add_edges_from([(0,1), (1,2), (2,3)])

    I = {0}
    O = {3}

    return G, I, O


def __edges_test(G: nx.Graph, m):
    for edge, node in zip(G.edges, G.nodes):
        i, j = edge
        if i % m == 1:
            print(f"({i}, {j}): {edge}  node: {node}")


if __name__ == "__main__":
    G = nx.Graph()
    G, I, O = deutsch_jozsa()
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()