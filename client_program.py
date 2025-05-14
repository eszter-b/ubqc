from __future__ import annotations

import math
from typing import Any, Dict, Generator

import networkx

from pydynaa import EventExpression

from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta


PI = math.pi


class ClientProgram(Program):
    PEER = "server"

    def __init__(
            self,
            num_qubits: int,
            phi: list,
            trap: bool,
            dummy: int,
            theta: list,
            r: list,
            tagged_state: str,
            dependencies: dict,
            graph: networkx.Graph,
            output: set
    ):
        self._num_qubits = num_qubits
        self._phi = phi
        self._trap = trap
        self._dummy = dummy
        self._theta = theta
        self._r = r
        self._tagged_state = tagged_state
        self._dependencies = dependencies
        self._graph = graph
        self._output = output

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="client_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._num_qubits,
        )

    def run(
            self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]
        csocket: ClassicalSocket = context.csockets[self.PEER]

        # Step 1: Alice prepares and sends qubits to Bob
        csocket.send_int(self._num_qubits)
        p = []

        # Remote state preparation
        for i in range(self._num_qubits):
            if not (self._trap and self._dummy == i + 1):
                q = epr_socket.create_keep()[0]
                q.rot_Z(angle=self._theta[i])
                p.append(q.measure())
            else:
                q = epr_socket.create_keep()[0]
                p.append(q.measure())

        yield from conn.flush()
        p = [int(i) for i in p]
        p_r = [self._r[i] ^ p[i] for i in range(len(p))]
        s = []

        # Step 2: Alice sends values of delta_i
        for i in range(self._num_qubits):

            phi_prime = self._phi[i]
            delta = phi_prime + p_r[i] * PI

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

                delta = phi_prime + p_r[i] * PI

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

            s.append(int(m ^ self._r[i]))

        measurement_outcome = ""
        for i in self._output:
            measurement_outcome += f"{s[i]}"

        return {"measurement_outcome": measurement_outcome}
