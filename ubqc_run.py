from __future__ import annotations

import math
import os
import matplotlib.pyplot as plt
import numpy as np

import netsquid as ns

from squidasm.run.stack.config import StackNetworkConfig, HeraldedLinkConfig, LinkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from client_program import ClientProgram
from server_program import ServerProgram

# Run Universal Blind Quantum Computation application.
PI = math.pi
PI_OVER_2 = math.pi / 2


def fail_rate(
    cfg: StackNetworkConfig,
    num_times: int,
    tagged_state: str = "00",
    phi: list = [0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI],
    theta: list = [0, 0, 0, 0, 0, 0, 0, 0],
    r: list = [0, 0, 0, 0, 0, 0, 0, 0],
) -> None:

    alice_program = ClientProgram(depth=4, wires=2, phi=phi, trap=False, dummy=0, theta=theta, r=r)
    bob_program = ServerProgram()
    client_results, server_results = run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=num_times)

    m7s = [result["m7"] for result in server_results]
    m8s = [result["m8"] for result in server_results]

    cntr = 0
    for m7, m8 in zip(m7s, m8s):
        tag = str(m7)+str(m8)
        if tagged_state == tag:
            cntr += 1

    fail_rate = 1 - cntr/num_times

    p = [result["p"] for result in client_results]
    m = [result["measurement"] for result in server_results]

    measured_different = 0
    for i in range(num_times):
        for j in range(8):
            if p[i][j] != m[i][j]:
                measured_different += 1
    fail_with_no_clalculation = measured_different/(num_times*8)
    print("fail_with_no_clalculation: ", fail_with_no_clalculation)

    print("number of runs: ", num_times)
    print(f"|{tagged_state}>: {cntr}")
    print("fail rate: ",fail_rate)



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

        alice_program = ClientProgram(depth=4, wires=2, phi=phi, trap=False, dummy=0, theta=theta, r=r)
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

if __name__ == "__main__":
    LogManager.set_log_level("WARNING")
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    config = "generic_config.yaml"
    #config = "config_nv.yaml"
    #config = "trapped_ion_config.yaml"
    cfg_file = os.path.join(os.path.dirname(__file__), config)
    cfg = StackNetworkConfig.from_file(cfg_file)
    num_times = 100

    tagged_state = "00"

    phi = [0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI]  # |00> tagged
    theta = [0, 0, 0, 0, 0, 0, 0, 0]
    r = [0, 0, 0, 0, 0, 0, 0, 0]

    #coherence_time_sweep(cfg, num_times=num_times, phi=phi, T='T1')
    #coherence_time_sweep(cfg, num_times=num_times, phi=phi, T='T2')

    fail_rate(cfg, num_times=num_times, tagged_state=tagged_state, phi=phi)

