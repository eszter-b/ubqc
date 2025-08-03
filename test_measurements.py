from __future__ import annotations

import matplotlib
from matplotlib import pyplot as plt
import pytest

matplotlib.use('Qt5Agg')
import os
import netsquid as ns

from squidasm.sim.stack.common import LogManager
from squidasm.run.stack.config import StackNetworkConfig

from flow import get_dependencies
import graph_state
import ubqc_run
import util
import numpy as np


LogManager.set_log_level("WARNING")

@pytest.fixture(scope='module')
def setup_env():
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "config/generic_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    num_times = 100
    tagged_state = "11"
    G, inputs, outputs = graph_state.grover_4_element()

    number_of_qubits = len(G.nodes)
    phi = util.get_phi_values(tagged_state)
    theta = util.generate_random_angles(num_qubits=number_of_qubits, is_blind=True)
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    dependency, correction = get_dependencies(G, phi, inputs, outputs)

    return {
        "cfg": cfg,
        "graph": G,
        "phi": phi,
        "theta": theta,
        "r": r,
        "num_times": num_times,
        "dependency": dependency,
        "correction": correction,
        "outputs": outputs,
        "tagged_state": tagged_state,
        "number_of_qubits": number_of_qubits
    }


@pytest.fixture(scope="session")
def plot_success_rate():
    success_rate = []
    yield success_rate

    if success_rate:
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        result_00 = success_rate[0]
        ax[0, 0].bar(result_00.keys(), result_00.get("00"))
        ax[0, 0].set_title(f"Tagged state: |00>")

        result_01 = success_rate[1]
        ax[0, 1].bar(result_01.keys(), result_01.get("01"))
        ax[0, 1].set_title(f"Tagged state: |01>")

        result_10 = success_rate[2]
        ax[1, 0].bar(result_10.keys(), result_10.get("10"))
        ax[1, 0].set_title(f"Tagged state: |10>")

        result_11 = success_rate[3]
        ax[1, 1].bar(result_11.keys(), result_11.get("11"))
        ax[1, 1].set_title(f"Tagged state: |11>")

        fig.suptitle("Success rates for ideal qdevice and chanel model for 100 runs.")
        fig.supxlabel("Tagged states")
        fig.supylabel("Success rates")
        fig.legend()
        plt.savefig(f"measurement_results/outcomes.png")


def test_graph_state():
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "config/generic_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    num_times = 1
    G, inputs, outputs = graph_state.brickwork(3,5)
    tagged_state = '000'

    number_of_qubits = len(G.nodes)
    phi = [0.0 for i in range(G.number_of_nodes())]
    theta = util.generate_random_angles(num_qubits=number_of_qubits, is_blind=True)
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    dependency, correction = get_dependencies(G, phi, inputs, outputs)
    success = ubqc_run.success_rate(
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
    assert success == 1


def test_deutsch_jozsa_balanced():
    """For a balanced function the expected success rate is 0.5"""

    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "config/generic_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    num_times = 250
    G, inputs, outputs = graph_state.deutsch_jozsa()
    tagged_state = '1'

    number_of_qubits = len(G.nodes)
    phi = [0.0, np.pi/2, np.pi, 0.0]
    theta = util.generate_random_angles(num_qubits=number_of_qubits, is_blind=True)
    theta[0] = 0.0
    theta[-1] = 0.0
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    dependency, correction = get_dependencies(G, phi, inputs, outputs)
    success = ubqc_run.success_rate(
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
    assert success == pytest.approx(0.5, 1e-1)


@pytest.mark.parametrize("tagged_state", ["00", "01", "10", "11"])
def test_success_rate(plot_success_rate, tagged_state):
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "config/generic_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    num_times = 100
    G, inputs, outputs = graph_state.grover_4_element()

    number_of_qubits = len(G.nodes)
    phi = util.get_phi_values(tagged_state)
    theta = util.generate_random_angles(num_qubits=number_of_qubits, is_blind=True)
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    dependency, correction = get_dependencies(G, phi, inputs, outputs)
    success = ubqc_run.success_rate(
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

    plot_success_rate.append(({tagged_state: success}))


def test_channel_fidelity_sweep(setup_env):
    ubqc_run.channel_fidelity_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        num_times=setup_env["num_times"],
        dependency=setup_env["dependency"],
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
    )


def test_channel_length_sweep(setup_env):
    ubqc_run.channel_length_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        num_times=10,
        dependency=setup_env["dependency"],
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        channel_length=np.arange(0, 200, step=1)
    )


def test_channel_length_sweep_trapped_ion(setup_env):
    config = "config/trapped_ions.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    ubqc_run.channel_length_sweep(
        cfg=cfg,
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        num_times=100,
        dependency=setup_env["dependency"],
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        channel_length=np.arange(0, 200, step=1)
    )


@pytest.mark.parametrize("noise", ["single_qubit_gate_depolar_prob", "two_qubit_gate_depolar_prob"])
def test_noise_sweep(setup_env, noise):
    ubqc_run.noise_model_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        noise=noise,
        dependency=setup_env["dependency"],
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        num_times=10
    )


def test_noise_sweep_client(setup_env):
    ubqc_run.noise_model_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        noise="two_qubit_gate_depolar_prob",
        dependency=setup_env["dependency"],
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        node=(0, 'client'),
        noise_prob=np.arange(start=0.0, stop=1.0, step=0.05),
        num_times=100
    )


def test_coherence_T1_sweep(setup_env):
    ubqc_run.coherence_time_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        dependency=setup_env["dependency"],
        coherence_times=np.logspace(start=1, stop=10, num=100, base=10, dtype=int),
        mode='T1',
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        num_times=100
    )


def test_coherence_T2_sweep(setup_env):
    ubqc_run.coherence_time_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        dependency=setup_env["dependency"],
        coherence_times=np.logspace(start=1, stop=10, num=50, base=10, dtype=int),
        mode='T2',
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        num_times=10
    )


def test_coherence_client_sweep(setup_env):
    ubqc_run.coherence_time_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        dependency=setup_env["dependency"],
        coherence_times=np.logspace(start=-3, stop=6, num=25, base=10, dtype=float),
        mode='T1',
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        node=(0, 'client'),
        num_times=10
    )