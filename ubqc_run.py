from __future__ import annotations

import math
import os
import matplotlib.pyplot as plt
import numpy as np

import netsquid as ns

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from client_program import ClientProgram
from server_program import ServerProgram

# Run Universal Blind Quantum Computation application.
PI = math.pi
PI_OVER_2 = math.pi / 2

def plot_fidelity(f7s, f8s):
    plt.subplot(1,2,1)
    plt.hist(f7s, color='skyblue')
    plt.legend(["qubit 6"])
    plt.ylabel("frequency")
    plt.xlabel("fidelity")
    plt.grid()
    plt.xticks(np.arange(0.7, 1.0, 0.1))

    plt.subplot(1,2,2)
    plt.hist(f8s, color='steelblue')
    plt.legend(["qubit 7"])
    plt.xlabel("fidelity")
    plt.grid()
    plt.xticks(np.arange(0.7, 1.0, 0.1))
    plt.suptitle("fidelities of output qubits for generic config")
    plt.savefig("fidelities_of_output_qubits_generic.png")

def trap_round(
    cfg: StackNetworkConfig,
    num_times: int = 1,
    depth: int = 4,
    wires: int = 2,
    phi: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    trap: bool =True,
    dummy: int = 1,
    theta: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    r: list = [0, 0, 0, 0, 0, 0, 0, 0],
) -> None:
    client_program = ClientProgram(
        depth = depth,
        wires = wires,
        phi = phi,
        trap = trap,
        dummy = dummy,
        theta = theta,
        r = r,
    )
    server_program = ServerProgram()

    client_results, server_results = run(
        cfg, {"client": client_program, "server": server_program}, num_times=num_times
    )

    #ps = [result["p"] for result in client_results]
    measurement = [result["measurement"] for result in server_results]
    ps = num_times*[int]
    ms = num_times*[int]
    for i in range(len(client_results)):
        ps[i] = client_results[i][dummy-1]
        ms[i] = measurement[i][dummy-1]
    num_fails = len([(p, m) for (p, m) in zip(ps, ms) if p != m])
    frac_fail = round(num_fails / num_times, 2)

    #print("ps: ", client_results)
    #print("ms: ", measurement)
    print(f"error rate: {frac_fail}")



if __name__ == "__main__":
    LogManager.set_log_level("WARNING")
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.GSLC)

    cfg_file = os.path.join(os.path.dirname(__file__), "generic_config.yaml")
    cfg = StackNetworkConfig.from_file(cfg_file)
    num_times = 1

    phi = [0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI]  # |00> tagged
    theta = [0, PI, 0, PI_OVER_2, 0, 0, 0, 0]
    r = [0, 0, 1, 0, 1, 0, 1, 0]
    #phi = [0, 0, 0, 0, 0, 0, 0, 0]
    #theta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    alice_program = ClientProgram(depth=4, wires=2, phi=phi, trap=False, dummy=0, theta=theta, r=r)
    bob_program = ServerProgram()
    _, server_results = run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=num_times)

    cntr = 0
    tagged_state = "00"

    m7s = [result["m7"] for result in server_results]
    m8s = [result["m8"] for result in server_results]
    for m7, m8 in zip(m7s, m8s):
        tag = str(m7)+str(m8)
        if tagged_state == tag:
            cntr += 1

    fail_rate = 1 - cntr/num_times

    f7s = [result["f7"] for result in server_results]
    f8s = [result["f8"] for result in server_results]
    sum_7 = np.sum(f7s)
    avg_7 = sum_7/num_times

    sum_8 = np.sum(f8s)
    avg_8 = sum_8/num_times

    print(f"average fidelities: q6: {avg_7}, q7: {avg_8}")

    #plot_fidelity(f7s, f8s)
    trap_round(cfg=cfg, num_times=1, dummy=2)

    print(f"num times: {num_times}\n|{tagged_state}>: {cntr}\nfail rate: {fail_rate}")
