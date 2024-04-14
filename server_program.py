from __future__ import annotations

from typing import Any, Dict, Generator

from netqasm.sdk.qubit import Qubit
from netqasm.lang.ir import BreakpointAction

from pydynaa import EventExpression
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

from brickwork_state import brickwork_graph


class ServerProgram(Program):
    PEER = "client"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="server_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=127,
        )

    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]
        csocket: ClassicalSocket = context.csockets[self.PEER]
        self.use_callbacks = True

        depth = yield from csocket.recv_int()
        wires = yield from csocket.recv_int()
        num_qubits = depth*wires

        if depth < wires:
            temp = wires
            wires = depth
            depth = temp

        #G = brickwork_graph(wires, depth)
        brickwork = [Qubit]
        measurement = []

        # Receive EPR Pairs
        brickwork = epr_socket.recv_keep(num_qubits)
        brickwork.reverse()     # reverse to match the qubit_id to place in list
        # Prepare brickwork state from received qubits
        brickwork[0].cphase(brickwork[2])
        brickwork[0].cphase(brickwork[1])
        brickwork[1].cphase(brickwork[2])
        brickwork[1].cphase(brickwork[3])
        """
        for edge in G.edges:
            k, l = edge
            brickwork[k-1].cphase(brickwork[l-1])
        """

        yield from conn.flush()
       
        print(f"number of received qubits: {len(brickwork)}")
        print(f"number of qubits expected: {num_qubits}")


        # Step 3: Measure and compute
        for i in range(len(brickwork)):
            # listen on csocket, progress if delta received
            delta = yield from csocket.recv_float()
            msg = yield from csocket.recv()
            assert msg == "delta sent"
            print(f"delta[{i}] at Bob: {delta}")

            brickwork[i].rot_Z(angle=delta)
            brickwork[i].H()
            conn.insert_breakpoint(BreakpointAction.DUMP_LOCAL_STATE)
            m = brickwork[i].measure(store_array=False)
            yield from conn.flush()
            csocket.send("qubit measured")
            measurement.append(int(m))
            print(f"measurement[{i}] at Bob: {measurement[i]}")

            csocket.send_int(measurement[i])

            
        print(f"number of measured qubits: {len(measurement)}")
        print(f"measurement results: {measurement}")

        return measurement
            