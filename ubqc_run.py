from __future__ import annotations

import matplotlib
matplotlib.use('Qt5Agg')
import math
import os
import matplotlib.pyplot as plt
from networkx import Graph
import numpy as np
import netsquid as ns

from examples.advanced.ubqc.util import linear_func, log_func
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from client_program import ClientProgram
from server_program import ServerProgram
from squidasm.run.stack.config import StackNetworkConfig, DepolariseLinkConfig, LinkConfig, HeraldedLinkConfig

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
    graph: Graph,
    theta: list,
    r: list,
    plot = False,
    **_kwargs
) -> float:

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
    bob_program = ServerProgram(graph=graph)
    client_results, server_results = run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=num_times)

    measurement_outcome = [result["measurement_outcome"] for result in client_results]
    success_count = sum(1 for m in measurement_outcome if m == tagged_state)
    success = success_count/num_times

    # Plot the distribution of outcomes for analysis
    if plot:
        outcome_counts = [m for m in measurement_outcome]
        unique, counts = np.unique(outcome_counts, return_counts=True)
        plt.bar(unique, counts)
        plt.xlabel("Final Measurement Outcomes")
        plt.ylabel("Frequency")
        plt.title(f"Outcome Distribution for Grover’s Amplified State |{tagged_state}>")
        plt.savefig(f"measurement_results/outcomes{tagged_state}.png")

    return success


def coherence_time_sweep(
    cfg: StackNetworkConfig,
    graph: Graph,
    dependency: dict,
    phi: list,
    theta: list,
    r: list,
    coherence_times: np.ndarray,
    num_times: int = 50,
    mode: str = 'T1',
    node: tuple = (1, 'server'),
    **kwargs
) -> None:

    if coherence_times is None:
        coherence_times = np.logspace(start=1, stop=10, num=50, base=10, dtype=int)

    node_key = node[0]
    if node[1] != cfg.stacks[node_key].name:
        return
    node_name = cfg.stacks[node_key].name

    results = []
    print(cfg.stacks[node_key].name)

    for coherence in coherence_times:
        if mode == 'T1':
            cfg.stacks[node_key].qdevice_cfg['T1'] = coherence
            cfg.stacks[node_key].qdevice_cfg['T2'] = coherence*0.1
            print(f"T1: {cfg.stacks[node_key].qdevice_cfg['T1']}    T2: {cfg.stacks[node_key].qdevice_cfg['T2']}")
        else:
            cfg.stacks[node_key].qdevice_cfg['T2'] = coherence
            cfg.stacks[node_key].qdevice_cfg['T1'] = 1e99
            print(cfg.stacks[node_key].qdevice_cfg['T2'])

        success = success_rate(
            cfg=cfg,
            num_times=num_times,
            phi=phi,
            dependency=dependency,
            graph=graph,
            r=r,
            theta=theta,
            **kwargs
        )
        results.append(success)

    plt.plot(coherence_times, results)
    plt.xscale('log', base=10)
    xlabel = f"Longitudinal relaxation time for {node_name} [ns]"
    if mode == 'T2':
        xlabel = f"Transverse relaxation time for {node_name} [ns]"

    coherence_times = np.array(coherence_times)
    results = np.array(results)

    y = 0.75
    x_cross = []
    crossings = np.where(np.diff(np.sign(results - y)))[0]
    if crossings.size > 0:
        i = crossings[0]
        # Interpolate to estimate the exact x
        x0, x1 = coherence_times[i], coherence_times[i + 1]
        y0, y1 = results[i], results[i + 1]
        x_interp = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        x_cross.append(x_interp)

    if len(x_cross) > 0:
        xc = float(x_cross[0])
        plt.plot(xc, y, 'ro')
        xc_mil = xc/1e6
        plt.axvline(x=xc, color='purple', linestyle='--', label=f'x ≈ {xc_mil:.4f} ms')

    plt.axhline(y=y, color='red', linestyle='--', label='y = 0.75')
    plt.xlabel(xlabel)
    plt.ylabel("Success rate")
    plt.grid()
    plt.legend()
    plt.savefig(f"measurement_results/{node_name}_{mode}_vs_success.png")



def noise_model_sweep(
        cfg: StackNetworkConfig,
        graph: Graph,
        dependency: dict,
        phi: list,
        theta: list,
        r: list,
        num_times: int = 100,
        noise: str = 'single_qubit_gate_depolar_prob',
        node: tuple = (1, 'server'),
        noise_prob: np.ndarray= None,
        **kwargs
) -> None:

    if noise_prob is None:
        noise_prob = np.arange(start=0.0, stop=0.05, step=0.001)

    results = []
    node_key = node[0]
    node_name = node[1]
    print(cfg.stacks[node_key])

    for n in noise_prob:
        if noise == 'single_qubit_gate_depolar_prob':
            cfg.stacks[node_key].qdevice_cfg['single_qubit_gate_depolar_prob'] = n
            print(cfg.stacks[node_key].qdevice_cfg['single_qubit_gate_depolar_prob'])
        else:
            cfg.stacks[node_key].qdevice_cfg['two_qubit_gate_depolar_prob'] = n
            print(cfg.stacks[node_key].qdevice_cfg['two_qubit_gate_depolar_prob'])


        success = success_rate(
            cfg=cfg,
            num_times=num_times,
            phi=phi,
            dependency=dependency,
            graph=graph,
            r=r,
            theta=theta,
            **kwargs
        )

        results.append(success)

    xlabel = f"Single qubit gate depolarisation probability for {node_name}"
    if noise == 'two_qubit_gate_depolar_prob':
        xlabel = f"Two qubit gate depolarisation probability for {node_name}"
    title = f"{node_key}_{noise}"
    util.fit_curve(results, noise_prob, xlabel, title, linear_func)


def channel_fidelity_sweep(
        cfg: StackNetworkConfig,
        graph: Graph,
        dependency: dict,
        phi: list,
        theta: list,
        r: list,
        num_times: int = 1,
        link_type: str = 'depolarise',
        **kwargs
) -> None:

    link_fidelity_list = np.arange(0.75, 1.0, step=0.01)

    results = []
    link_config = DepolariseLinkConfig.from_file("config/depolarise_link_config.yaml")
    link = LinkConfig(stack1="client", stack2="server", typ="depolarise", cfg=link_config)

    # Replace link from YAML file with new depolarise link
    cfg.links = [link]
    channel_type = cfg.links[0].typ
    print(channel_type)

    for fidelity in link_fidelity_list:
        link_config.fidelity = fidelity

        print(cfg.links[0].cfg)

        success = success_rate(
            cfg=cfg,
            num_times=num_times,
            phi=phi,
            dependency=dependency,
            graph=graph,
            r=r,
            theta=theta,
            **kwargs
        )

        results.append(success)

    xlabel = f"Fidelity of the {channel_type} quantum channel"

    util.fit_curve(results, link_fidelity_list, xlabel, f"{link_type}_link_fidelity")


def channel_length_sweep(
        cfg: StackNetworkConfig,
        graph: Graph,
        dependency: dict,
        phi: list,
        theta: list,
        r: list,
        channel_length: np.ndarray | None,
        num_times: int = 100,
        **kwargs
) -> None:

    if channel_length is None:
        channel_length = np.arange(0, 200, step=1)

    results = []
    link_config = HeraldedLinkConfig.from_file("config/heralded_link_config.yaml")

    link = LinkConfig(stack1="client", stack2="server", typ="heralded", cfg=link_config)

    # Replace link from YAML file with new depolarise link
    cfg.links = [link]
    channel_type = cfg.links[0].typ
    print(channel_type)

    for distance in channel_length:
        link_config.length = distance

        print(cfg.links[0].cfg)

        success = success_rate(
            cfg=cfg,
            num_times=num_times,
            phi=phi,
            dependency=dependency,
            graph=graph,
            r=r,
            theta=theta,
            **kwargs
        )

        results.append(success)

    xlabel = f"Length of the {channel_type} quantum channel [km]"
    util.fit_curve(results, channel_length, xlabel, f"heralded_link_", log_func)


if __name__ == "__main__":
    LogManager.set_log_level("WARNING")
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "config/generic_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    num_times = 100
    tagged_state = "11"
    G = fixed_graph_2_bit()

    number_of_qubits = len(G.nodes)
    phi = util.get_phi_values(tagged_state)
    theta = util.generate_random_angles(num_qubits=number_of_qubits, is_blind=True)
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    inputs = {0, 1}
    outputs = {8, 9}
    dependency, correction = get_dependencies(G, phi, inputs, outputs)

    success = success_rate(
        cfg=cfg,
        num_times=num_times,
        number_of_qubits=number_of_qubits,
        tagged_state=tagged_state,
        phi=phi,
        dependency=dependency,
        graph=G,
        outputs=outputs,
        r=r,
        theta=theta,
        output=outputs,
        plot=False
    )

    util.print_results_default(success, num_times, tagged_state)

