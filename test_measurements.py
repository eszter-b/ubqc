from __future__ import annotations

import matplotlib
import pytest

matplotlib.use('Qt5Agg')
import os
import netsquid as ns

from squidasm.sim.stack.common import LogManager
from squidasm.run.stack.config import StackNetworkConfig

from flow import get_dependencies
from brickwork_state import fixed_graph_2_bit
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

    num_times = 10
    tagged_state = "11"
    G = fixed_graph_2_bit()

    number_of_qubits = len(G.nodes)
    phi = util.get_phi_values(tagged_state)
    theta = util.generate_random_angles(num_qubits=number_of_qubits, is_blind=True)
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    inputs = {0, 1}
    outputs = {8, 9}
    dependency, correction = get_dependencies(G, phi, inputs, outputs)

    return {
        "cfg": cfg,
        "graph": G,
        "phi": phi,
        "theta": theta,
        "r": r,
        "num_times": 10,
        "dependency": dependency,
        "correction": correction,
        "outputs": outputs,
        "tagged_state": tagged_state,
        "number_of_qubits": number_of_qubits
    }


@pytest.mark.parametrize("tagged_state", ["00", "01", "10", "11"])
def test_success_rate(tagged_state):
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "config/generic_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)

    num_times = 100
    G = fixed_graph_2_bit()

    number_of_qubits = len(G.nodes)
    phi = util.get_phi_values(tagged_state)
    theta = util.generate_random_angles(num_qubits=number_of_qubits, is_blind=True)
    r = util.generate_random_key(num_qubits=number_of_qubits, is_blind=True)

    inputs = {0, 1}
    outputs = {8, 9}
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
    assert success == 1.0


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

def test_coherence_client_sweep(setup_env):
    ubqc_run.coherence_time_sweep(
        cfg=setup_env["cfg"],
        graph=setup_env["graph"],
        phi=setup_env["phi"],
        theta=setup_env["theta"],
        r=setup_env["r"],
        dependency=setup_env["dependency"],
        coherence_times=np.logspace(start=-3, stop=6, num=25, base=10, dtype=float),
        mode='T2',
        output=setup_env["outputs"],
        tagged_state=setup_env["tagged_state"],
        number_of_qubits=setup_env["number_of_qubits"],
        node={1: 'server'},
        num_times=10
    )