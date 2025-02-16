from __future__ import annotations

import math
import os
import matplotlib.pyplot as plt
from networkx import Graph
import numpy as np
from typing import List

import netsquid as ns

from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from client_program import ClientProgram
from server_program import ServerProgram
from squidasm.run.stack.config import (
    GenericQDeviceConfig,
    LinkConfig,
    StackConfig,
    StackNetworkConfig,
)

from flow import get_dependencies
from brickwork_state import fixed_graph_2_bit

# Run Universal Blind Quantum Computation application.
PI = math.pi
PI_OVER_2 = math.pi / 2


def fail_rate(
    cfg: StackNetworkConfig,
    num_times: int,
    tagged_state: str,
    phi: list,
    dependency: dict,
    graph: Graph,
    theta: list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    r: list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
) -> None:

    alice_program = ClientProgram(depth=5, wires=2, phi=phi, trap=False, dummy=0, theta=theta, r=r, tagged_state=tagged_state, dependencies=dependency, graph=graph)
    bob_program = ServerProgram()
    client_results, server_results = run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=num_times)

    m8s = [result["m8"] for result in server_results]
    m9s = [result["m9"] for result in server_results]
    success_count = sum(1 for m8, m9 in zip(m8s, m9s) if f"{m8}{m9}" == tagged_state)

    # Calculate and print the fail rate
    fail_rate = 1 - success_count / num_times
    print("Number of runs:", num_times)
    print(f"Number of |{tagged_state}> outcomes:", success_count)
    print("Fail rate:", fail_rate)

    # Optionally, plot the distribution of outcomes for analysis
    outcome_counts = [f"{m8}{m9}" for m8, m9 in zip(m8s, m9s)]
    unique, counts = np.unique(outcome_counts, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel("Final Measurement Outcomes")
    plt.ylabel("Frequency")
    plt.title(f"Outcome Distribution for Grover’s Amplified State |{tagged_state}>")
    plt.grid()
    plt.savefig("fail_rate.png")


def coherence_time_sweep(
    cfg: StackNetworkConfig,
    num_times: int = 1,
    phi: list = [0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI],
    theta: list = [0, 0, 0, 0, 0, 0, 0, 0],
    r: list = [0, 0, 1, 0, 1, 0, 1, 0],
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

        alice_program = ClientProgram(depth=4, wires=2, phi=phi, trap=False, dummy=0, theta=theta, r=r, tagged_state="00")
        bob_program = ServerProgram()
        _, server_results = run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=num_times)

        m7s = [result["m7"] for result in server_results]
        m8s = [result["m8"] for result in server_results]

        cntr = 0
        for m7, m8 in zip(m7s, m8s):
            tag = str(m7)+str(m8)
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



def get_average(num_times, fs):
    sum = np.sum(fs)
    avg = sum/num_times
    return avg

def create_network(
    node_names: List[str] = None, link_noise: float = 0, qdevice_noise: float = 0
) -> StackNetworkConfig:

    node_names = ["client", "server"] if node_names is None else node_names
    assert len(node_names) == 2

    qdevice_cfg = GenericQDeviceConfig.perfect_config()
    qdevice_cfg.num_qubits = 8
    stacks = [
        StackConfig(name=name, qdevice_typ="generic", qdevice_cfg=qdevice_cfg)
        for name in node_names
    ]

    link = LinkConfig.perfect_config(
        stack1=node_names[0], stack2=node_names[1]
    )
    return StackNetworkConfig(stacks=stacks, links=[link])


if __name__ == "__main__":
    LogManager.set_log_level("WARNING")
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "generic_config.yaml"
    #config = "config_nv.yaml"
    #config = "trapped_ion_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)
    """
    alice = cfg.stacks[0].qdevice_cfg
    bob = cfg.stacks[1].qdevice_cfg
    alice = GenericQDeviceConfig.perfect_config()
    bob= GenericQDeviceConfig.perfect_config()
    alice.num_qubits = 8
    bob.num_qubits = 8
    """
    #cfg = create_network(node_names=["client", "server"])
    num_times = 100

    tagged_state = "00"

    #phi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # no calculation
    phi = [0.0, 0.0, 0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI]
    theta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #coherence_time_sweep(cfg, num_times=num_times, phi=phi, T='T1')
    #coherence_time_sweep(cfg, num_times=num_times, phi=phi, T='T2')
    inputs = {0, 1}
    outputs = {8, 9}
    G = fixed_graph_2_bit()

    dependency, correction = get_dependencies(G, phi, inputs, outputs)

    fail_rate(cfg, num_times=num_times, tagged_state=tagged_state, phi=phi, dependency=dependency, graph=G)

