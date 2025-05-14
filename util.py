import numpy as np
import random
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

from squidasm.run.stack.config import (
    GenericQDeviceConfig,
    LinkConfig,
    StackConfig,
    StackNetworkConfig,
)

PI = np.pi
MIN_SUCCESS_RATE = 0.75


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


def print_results_default(success_rate, num_times, tagged_state):
    print("Number of runs:", num_times)
    print(f"Number of |{tagged_state}> outcomes:", int(success_rate*num_times))
    print("Success rate:", success_rate)


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


def linear_func(x, m, c):
    return m * x + c

def log_func(x, A, L, k, x0):
    return A - (L / (1 + np.exp(-k * (x - x0))))


def find_x_for_y(func, params, y_target, x0):
    x_target, = fsolve(lambda x: func(x, *params) - y_target, x0)
    return x_target


def fit_curve(results: list, x: np.ndarray, xlabel: str, title: str, model: callable=exp_func, show: bool=False) -> None:

    params, covariance = curve_fit(model, x, results)

    # Generate x-values for a smooth curve
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = model(x_fit, *params)
    x0 = (x[0] + x[-1]) / 2
    x_target = find_x_for_y(model, params, MIN_SUCCESS_RATE, x0)

    print("x value for y = {:.2f} is {:.4f}".format(MIN_SUCCESS_RATE, x_target))

    # Plot the original data and the fitted curve
    plt.scatter(x, results, label='Measurement results')
    plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')

    # Plot horizontal and vertical lines at y_target and x_target
    plt.axhline(y=MIN_SUCCESS_RATE, color='gray', linestyle='--', label=f'y = {MIN_SUCCESS_RATE}')
    plt.axvline(x=x_target, color='purple', linestyle='--', label=f'x ≈ {x_target:.4f}')

    # Highlight the point on the fitted curve
    plt.scatter(x_target, MIN_SUCCESS_RATE, color='green', zorder=5, label='Target Point')

    plt.xlabel(xlabel)
    plt.ylabel('Success rate')
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(f"measurement_results/{title}_vs_success.png")

