from __future__ import annotations

import math
import os
import matplotlib.pyplot as plt
from networkx import Graph
import numpy as np
from typing import List

import netsquid as ns

from examples.advanced.ubqc.brickwork_state import triangular_cluster, fixed_graph_small
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from client_program import ClientProgram
from server_program import ServerProgram
from squidasm.run.stack.config import StackNetworkConfig

from flow import get_dependencies
from brickwork_state import fixed_graph_2_bit
import util


# Run Universal Blind Quantum Computation application.
PI = math.pi
PI_OVER_2 = math.pi / 2


def success_rate(
    cfg: StackNetworkConfig,
    num_times: int,
    number_of_qubits: int,
    tagged_state: str,
    phi: list,
    dependency: dict,
    output: set,
    outputs: set,
    graph: Graph,
    theta: list,
    r: list,
) -> None:

    alice_program = ClientProgram(
        num_qubits=number_of_qubits,
        phi=phi,
        trap=False,
        dummy=0,
        theta=theta,
        r=r,
        tagged_state=tagged_state,
        dependencies=dependency,
        graph=graph,
        output=output
    )
    bob_program = ServerProgram(graph=graph, outputs=outputs)
    client_results, server_results = run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=num_times)

    measurement_outcome = [result["measurement_outcome"] for result in client_results]
    #m9s = [result["m9"] for result in server_results]
    success_count = sum(1 for m in measurement_outcome if m == tagged_state)

    # Calculate and print the fail rate

    print("Number of runs:", num_times)
    print(f"Number of |{tagged_state}> outcomes:", success_count)
    print("Success rate:", success_count / num_times)

    # Optionally, plot the distribution of outcomes for analysis
    outcome_counts = [m for m in measurement_outcome]
    unique, counts = np.unique(outcome_counts, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel("Final Measurement Outcomes")
    plt.ylabel("Frequency")
    plt.title(f"Outcome Distribution for Grover’s Amplified State |{tagged_state}>")
    plt.grid()
    plt.savefig("outcomes.png")


def coherence_time_sweep(
    cfg: StackNetworkConfig,
    graph: Graph,
    phi: list,
    theta: list,
    r: list,
    num_times: int = 1,
    T: str = 'T1'
    ) -> None:

    T1_coherence = np.arange(start=1000000000, stop=100000000000, step=2500000000)
    T2_coherence = np.arange(start=100000000, stop=10000000000, step=250000000)

    sweep = []
    if T == 'T1':
        sweep = T1_coherence
    elif T == 'T2':
        sweep = T2_coherence
    else:
        print("invalid parameter for coherence time")
        return

    success_rate = []

    for coherence in sweep:
        if T == 'T1':
            cfg.stacks[1].qdevice_cfg['T1'] = int(coherence)
        else:
            cfg.stacks[1].qdevice_cfg['T2'] = int(coherence)

        alice_program = ClientProgram(
            num_qubits=number_of_qubits,
            phi=phi,
            trap=False,
            dummy=0,
            theta=theta,
            r=r,
            tagged_state=tagged_state,
            dependencies=dependency,
            graph=graph,
            output=outputs
        )

        bob_program = ServerProgram(graph=G, outputs=outputs)
        _, server_results = run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=num_times)

        m8s = [result["m8"] for result in server_results]
        m9s = [result["m9"] for result in server_results]

        cntr = 0
        for m8, m9 in zip(m8s, m9s):
            tag = str(m8)+str(m9)
            if tagged_state == tag:
                cntr += 1

        success = cntr/num_times
        success_rate.append(success)

    plt.plot(sweep, success_rate)
    if T == 'T1':
        plt.xlabel("Longitudinal relaxation time")
    else:
        plt.xlabel("Transverse relaxation time")

    plt.ylabel("Success rate")
    plt.grid()
    plt.savefig("UBQC_"+T+"_vs_success.png")



if __name__ == "__main__":
    LogManager.set_log_level("WARNING")
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "generic_config.yaml"
    #config = "config_nv.yaml"
    #config = "trapped_ion_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    num_times = 100
    tagged_state = "10"
    G = fixed_graph_2_bit()

    number_of_qubits = len(G.nodes)
    phi = util.get_phi_values(tagged_state)
    theta = util.generate_random_angles(num_qubits=number_of_qubits)
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    #coherence_time_sweep(cfg, num_times=num_times, phi=phi, T='T1')
    #coherence_time_sweep(cfg, num_times=num_times, phi=phi, T='T2')
    inputs = {0, 1}
    outputs = {8, 9}
    dependency, correction = get_dependencies(G, phi, inputs, outputs)

    """
    A: 00, 10
    B: 01, 11   (01 is faulty still)
    """

    success_rate(
        cfg,
        num_times=num_times,
        number_of_qubits=number_of_qubits,
        tagged_state=tagged_state,
        phi=phi,
        dependency=dependency,
        graph=G,
        outputs=outputs,
        r=r,
        theta=theta,
        output=outputs
    )

