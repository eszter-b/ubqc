from __future__ import annotations

from typing import Any, Dict, Generator

import networkx
from netqasm.sdk.qubit import Qubit

from netsquid.qubits.dmutil import dm_fidelity
from netsquid.qubits import ketstates, ketutil

from pydynaa import EventExpression

from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state


class ServerProgram(Program):
    PEER = "client"
    def __init__(
            self,
            graph: networkx.Graph
    ):
        self._graph = graph
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="server_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=10,
        )

    def run(
            self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]
        csocket: ClassicalSocket = context.csockets[self.PEER]
        self.use_callbacks = True

        num_qubits = yield from csocket.recv_int()

        brickwork = [Qubit]
        measurement = []
        fidelity = []

        # Receive EPR Pairs
        for i in range(num_qubits):
            if i == 0:
                brickwork[0] = epr_socket.recv_keep()[0]
            else:
                brickwork.append(epr_socket.recv_keep()[0])


        yield from conn.flush()

        # Prepare brickwork state from received qubits
        for edge in self._graph.edges:
            k, l = edge
            brickwork[k].cphase(brickwork[l])
        yield from conn.flush()

        # Step 3: Measure and compute
        for i in range(num_qubits):

            # listen on csocket, progress if delta received
            delta = yield from csocket.recv_float()
            msg = yield from csocket.recv()
            assert msg == "delta sent"
            csocket.send("delta received")

            brickwork[i].H()
            brickwork[i].rot_Z(angle=delta)
            brickwork[i].H()

            yield from conn.flush()
            msg = yield from csocket.recv()
            assert msg == "ping"

            epr_server = get_qubit_state(brickwork[i], "server")

            fidelity.append(float(
                dm_fidelity(epr_server, ketutil.reduced_dm(ketstates.b00, [0]))
            ))
            csocket.send("pong")

            m = brickwork[i].measure()
            yield from conn.flush()
            csocket.send("qubit measured")
            measurement.append(int(m))
            csocket.send_int(measurement[i])

        return {"measurement": measurement, "m8": measurement[-2], "m9": measurement[-1], "fidelity": fidelity}
