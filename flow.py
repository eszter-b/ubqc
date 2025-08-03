import graphix
from networkx import Graph
from graphix.command import CommandKind
import graph_state
import numpy as np


PI = np.pi

def get_dependencies(graph: Graph, angles: list, inputs: set, outputs: set, draw_pattern: bool = False):
    meas_angles = {}
    for i in range(len(angles)):
        meas_angles.update({i: angles[i]})

    pattern = graphix.generator.generate_from_graph(graph, angles=meas_angles, inputs=inputs, outputs=outputs)

    pattern.standardize()
    pattern.shift_signals()
    if draw_pattern:
        pattern.draw_graph(flow_from_pattern=False, show_measurement_planes=False, show_local_clifford=True)

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
    G, inputs, outputs = graph_state.deutsch_jozsa()
    #phi = [0.0, 0.0, 0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI]
    phi = [0.0 for i in range(G.number_of_nodes())]
    print(phi)

    dependency, correction = get_dependencies(G, phi, inputs, outputs, True)
    print(dependency)
    print(correction)
