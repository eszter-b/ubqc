from __future__ import annotations

from typing import Any, Dict, Generator

from netqasm.sdk.qubit import Qubit

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

        brickwork = [Qubit]
        measurement = []

        # Receive EPR Pairs
        brickwork = epr_socket.recv_keep(num_qubits)
        yield from conn.flush()
        brickwork.reverse()     # reverse to match the qubit_id to place in list

        print(f"number of received qubits: {len(brickwork)}")
        print(f"number of qubits expected: {num_qubits}")

        G = brickwork_graph(wires, depth)

        # Step 3: Measure and compute
        for q, i in zip(brickwork, range(len(brickwork))):
            # listen on csocket, progress if delta received
            delta = yield from csocket.recv_float()
            msg = yield from csocket.recv()
            assert msg == "delta sent"
            #csocket.send("delta arrived")
            print(f"delta[{i}] at Bob: {delta}")
            E = G.edges(i+1)

            # Prepare brickwork state from received qubits
            for edge in E:
                k, l = edge
                brickwork[k-1].cphase(brickwork[l-1])

            q.rot_Z(angle=delta)
            q.H()
            #conn.insert_breakpoint(BreakpointAction.DUMP_LOCAL_STATE)
            m = q.measure(store_array=False)
            yield from conn.flush(block=False, callback="conn yielded")
            csocket.send("qubit measured")
            measurement.append(int(m))
            print(f"measurement[{i}] at Bob: {measurement[i]}")

            csocket.send_int(measurement[i])

            
        print(f"number of measured qubits: {len(measurement)}")
        print(f"measurement results: {measurement}")

        return measurement
            