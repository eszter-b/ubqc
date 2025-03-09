import numpy as np
import random

from squidasm.run.stack.config import (
    GenericQDeviceConfig,
    LinkConfig,
    StackConfig,
    StackNetworkConfig,
)

PI = np.pi


def generate_random_angles(num_qubits, is_blind=False) -> list[float]:
    """
    Populates the given list with random angles that are multiples of pi/4
    within the range [0, 7*pi/4].
    """
    theta = []
    if is_blind:
        possible_angles = [i * (PI / 4) for i in range(8)]
        for i in range(num_qubits):
            theta.append(random.choice(possible_angles))
    else:
        for i in range(num_qubits):
            theta.append(0.0)

    return theta


def generate_random_key(num_qubits, is_blind=False) -> list[int]:
    """
    Populates the given list with random key where r[i]: {0, 1}.
    """
    r = []

    if is_blind:
        possible_bits = [0, 1]
        for i in range(num_qubits):
            r.append(random.choice(possible_bits))
    else:
        for i in range(num_qubits):
            r.append(0)

    return r


def get_phi_values(tagged_state: str) -> list[float]:
    """
    Returns the phi values based on the given tagged_state.

    Parameters:
        tagged_state (str): A binary string representing the tagged state ("00", "01", "10", "11").

    Returns:
        list[float]: A list of phi values corresponding to the given tagged_state.
    """

    phi_map = {
        "00": [0.0, 0.0, 0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI],
        "01": [0.0, 0.0, 0.0, 0.0, 0.0, PI, 0.0, 0.0, PI, PI],
        "10": [0.0, 0.0, 0.0, 0.0, PI, 0.0, 0.0, 0.0, PI, PI],
        "11": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, PI, PI],
        "no_calc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    # Default value if tagged_state is not found
    return phi_map.get(tagged_state, [0.0] * 8)



def get_average(num_times, fs):
    sum = np.sum(fs)
    avg = sum/num_times
    return avg

def create_network(
    node_names: list[str] = None, link_noise: float = 0, qdevice_noise: float = 0
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