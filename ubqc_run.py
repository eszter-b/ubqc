from __future__ import annotations

import math
import os

import netsquid as ns

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from client_program import ClientProgram
from server_program import ServerProgram

# Run Universal Blind Quantum Computation application.
PI = math.pi
PI_OVER_2 = math.pi / 2


if __name__ == "__main__":
    LogManager.set_log_level("WARNING")
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)

    cfg_file = os.path.join(os.path.dirname(__file__), "config_nv.yaml")
    cfg = StackNetworkConfig.from_file(cfg_file)

    phi = [0.0, 0.0, PI, PI, 0.0, 0.0, PI, PI]  # |00> tagged
    theta = [0, 0, 0, 0, 0, 0, 0, 0]
    r = [0, 0, 0, 0, 0, 0, 0, 0]
    #phi = [PI/4, PI/4, PI/4, 0, 0, 0, 0, 0, 0, 0]
    #theta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    """ 
        H gate: phi0 = pi/4    phi1 = pi/4     phi2 = pi/4
        T gate: phi0 = pi/8    phi1 = 0        phi2 = 0
        I gate: phi0 = 0       phi1 = 0        phi2 = 0
    """

    alice_program = ClientProgram(depth=4, wires=2, phi=phi, trap=False, dummy=0, theta=theta, r=r)
    bob_program = ServerProgram()
    run(config=cfg, programs={"client": alice_program, "server": bob_program}, num_times=1)


