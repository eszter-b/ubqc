from __future__ import annotations

import math
from typing import Any, Dict, Generator

from pydynaa import EventExpression
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.routines import remote_state_preparation

from brickwork_state import fixed_graph_2_bit


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
    ):
        self._depth = depth
        self._wires = wires
        self._phi = phi
        self._trap = trap
        self._dummy = dummy
        self._theta = theta
        self._r = r

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="client_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._depth*self._wires,
        )

    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]
        csocket: ClassicalSocket = context.csockets[self.PEER]

        #Step 1: Alice prepares and sends qubits to Bob
        num_qubits = self._depth*self._wires
        csocket.send_int(self._depth)
        csocket.send_int(self._wires)
        p = []

        # Remote state preparation
        for i in range(num_qubits):
            if not (self._trap and self._dummy == i+1):
                p.append(remote_state_preparation(epr_socket, self._theta[i]))
            else:
                p.append(remote_state_preparation(epr_socket, 0))

        yield from conn.flush()
        p = [int(i) for i in p]

        s = []
        G = fixed_graph_2_bit()
        wire_0 = []
        wire_1 = []
        for i in range(7):
            wire = G.nodes[i]['pos']
            x, y = wire
            E = G.edges(i)
            for edge in E:
                k, l = edge
                int(k)
                int(l)
                if y==0:
                    wire_0.append(k)
                    wire_1.append(0)
                else:
                    wire_1.append(k)
                    wire_0.append(0)

        # Step 2: Alice sends values of delta_i
        for i in range(num_qubits):
            wire = G.nodes[i]['pos']
            x, y = wire
            j = 0
            p_r = p[i] + self._r[i]

            delta = -self._theta[i] + p_r*PI
            phi_prime = self._phi[i]

            if not (self._trap and self._dummy == i+1):
                if i == 2 or i == 3:
                    if y == 0:
                        j = wire_0[i]
                    else:
                        j = wire_1[i]
                    phi_prime = math.pow(-1, s[j])*self._phi[i]
                elif i > 3:
                    if y == 0:
                        j = wire_0[i-2]
                        x = wire_0[i-3]
                    else:
                        j = wire_1[i]
                        x = wire_1[i-3]
                    phi_prime = math.pow(-1, s[j])*self._phi[i] + s[x]*PI

                delta = phi_prime - self._theta[i] + p_r*PI

            csocket.send_float(delta)
            csocket.send("delta sent")
            msg = yield from csocket.recv()
            assert msg == "delta recieved"

            # Wait for fidelity measurement at Bob
            csocket.send("ping")
            msg = yield from csocket.recv()
            assert msg == "pong"

            # Proceed if the qubit is measured at Bob and apply correction to measurement
            msg = yield from csocket.recv()
            assert msg == "qubit measured"
            m = yield from csocket.recv_int()
            s.append(abs(int(m)-self._r[i]))

        return {"p": p}
    