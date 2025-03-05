from __future__ import annotations

import math
from typing import Any, Dict, Generator

import networkx

from pydynaa import EventExpression

from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.routines import remote_state_preparation


PI = math.pi


class ClientProgram(Program):
    PEER = "server"

    def __init__(
            self,
            depth: int,
            wires: int,
            phi: list,
            trap: bool,
            dummy: int,
            theta: list,
            r: list,
            tagged_state: str,
            dependencies: dict,
            graph: networkx.Graph
    ):
        self._depth = depth
        self._wires = wires
        self._phi = phi
        self._trap = trap
        self._dummy = dummy
        self._theta = theta
        self._r = r
        self._tagged_state = tagged_state
        self._dependencies = dependencies
        self._graph = graph

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="client_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._depth * self._wires,
        )

    def run(
            self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]
        csocket: ClassicalSocket = context.csockets[self.PEER]

        # Step 1: Alice prepares and sends qubits to Bob
        num_qubits = self._depth * self._wires
        csocket.send_int(self._depth)
        csocket.send_int(self._wires)
        p = []

        # Remote state preparation
        for i in range(num_qubits):
            if not (self._trap and self._dummy == i + 1):
                #q = epr_socket.create_keep()[0]
                # q.Z()
                # q.X()
                # p.append(measXY(q, self._theta[i]))
                p.append(remote_state_preparation(epr_socket, self._theta[i]))
            else:
                p.append(remote_state_preparation(epr_socket, 0))

        yield from conn.flush()
        p = [int(i) for i in p]
        #print("Alice: ", p)
        s = []

        # Step 2: Alice sends values of delta_i
        for i in range(num_qubits):

            p_r = self._r[i] ^ p[i]
            phi_prime = self._phi[i]
            delta = phi_prime - self._theta[i] + p_r * PI

            if not (self._trap and self._dummy == i + 1):

                node = self._dependencies[i]
                s_dependency = [s for s in node['s']]
                t_dependency = [t for t in node['t']]
                if s_dependency != 0:
                    x = sum([s[i] for i in s_dependency])%2
                    phi_prime *= math.pow(-1, x)
                if t_dependency != 0:
                    z = sum([s[i] for i in t_dependency])%2
                    phi_prime += z * PI

                delta = phi_prime - self._theta[i] + p_r * PI
                """
                if i == 4:
                    phi_prime = math.pow(-1, s[2])*self._phi[i] + s[1]*PI
                elif i == 5:
                    phi_prime = math.pow(-1, s[3])*self._phi[i] + s[0] * PI
                elif i == 6:
                    phi_prime = s[2] * PI
                elif i == 7:
                    phi_prime = s[3] * PI
                elif i == 8:
                    phi_prime = math.pow(-1, s[7])*self._phi[i] + s[5] * PI
                elif i == 9:
                    phi_prime = math.pow(-1, s[6])*self._phi[i] + s[4] * PI
                delta = phi_prime - self._theta[i] + p_r * PI
                """
            csocket.send_float(delta)
            csocket.send("delta sent")
            msg = yield from csocket.recv()
            assert msg == "delta received"

            # Wait for fidelity measurement at Bob
            csocket.send("ping")
            msg = yield from csocket.recv()
            assert msg == "pong"

            # Proceed if the qubit is measured at Bob and apply correction to measurement
            msg = yield from csocket.recv()
            assert msg == "qubit measured"
            m = yield from csocket.recv_int()

            s.append(int(m))

        return {"p": p}
