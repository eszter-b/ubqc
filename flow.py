import graphix
from networkx import Graph
from graphix.command import CommandKind
from brickwork_state import fixed_graph_2_bit
import numpy as np


PI = np.pi

def get_dependencies(graph: Graph, angles: list, inputs: set, outputs: set):
    meas_angles = {}
    for i in range(len(angles)):
        meas_angles.update({i: angles[i]})

    pattern = graphix.generator.generate_from_graph(graph, angles=meas_angles, inputs=inputs, outputs=outputs)

    pattern.standardize()
    pattern.shift_signals()

    dependency_set = {}
    correction_set = {}
    correction_X = {}
    correction_Z = {}

    pattern_seq = pattern._Pattern__seq
    for c in pattern_seq:
        if c.kind == CommandKind.M:
            dependency_set.update({c.node: {"s": c.s_domain, "t": c.t_domain}})
        if c.kind == CommandKind.X:
            correction_X.update(({c.node: {"s": c.domain}}))
        if c.kind == CommandKind.Z:
            correction_Z.update(({c.node: {"t": c.domain}}))

    for node in correction_X:
        dependency_set.update({node: {"s": correction_X[node]["s"], "t" : correction_Z[node]["t"]}})
        correction_set.update({node: {"s": correction_X[node]["s"], "t" : correction_Z[node]["t"]}})

    return dependency_set, correction_set


if __name__ == "__main__":
    G = fixed_graph_2_bit()
    phi = [0.0, 0.0, 0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI]

    inputs = {0, 1}
    outputs = {8, 9}

    dependency, correction = get_dependencies(G, phi, inputs, outputs)
    print(dependency)
    print(correction)
