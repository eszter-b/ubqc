from __future__ import annotations

import math
from typing import Any, Dict, Generator

from pydynaa import EventExpression
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta


PI = math.pi
PI_OVER_2 = math.pi / 2

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
        epr = epr_socket.create_keep(num_qubits)
        p = []

        # Remote state preparation
        for q, i in zip(epr, range(num_qubits-1, -1, -1)):
            if not (self._trap and self._dummy == i+1):
                q.rot_Z(angle=self._theta[i])
                q.H()
            m = q.measure()
            p.append(m)
        yield from conn.flush()
        p = [int(i) for i in p]
        print(f"p: {p}")

        mesurement = []

        # Step 2: Alice sends values of delta_i
        for i in range(len(self._theta)):
            if self._trap and self._dummy == i+1:
                delta = -self._theta[i] + (p[i] + self._r[i]) * PI
            elif i == 0:
                delta = self._theta[i] - self._phi[i] + self._r[i] * PI
            else:
                delta = (
                    math.pow(-1, (mesurement[i-1] + self._r[i-1])) * self._phi[i]
                    - self._theta[i]
                    + (p[i] + self._r[i]) * PI
                )
            print(f"delta[{i}] at Alice: {delta}")

            csocket.send_float(delta)
            csocket.send("delta sent")

            # Proceed if the qubit is measured at Bob
            msg_measurement = yield from csocket.recv()
            assert msg_measurement == "qubit measured"

            m = yield from csocket.recv_int()
            mesurement.append(int(m))
            print(f"mesurement[{i}] at Alice: {mesurement[i]}")
            
        print(f"number of qubits sent: {len(p)}")
        print(f"tagged state: |{mesurement[-2]}{mesurement[-1]}>")

        return p
