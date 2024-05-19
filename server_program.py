from __future__ import annotations

from typing import Any, Dict, Generator

from netqasm.sdk.qubit import Qubit

from netsquid.qubits.dmutil import dm_fidelity
from netsquid.qubits import ketstates, ketutil

from pydynaa import EventExpression
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.routines import recv_remote_state_preparation
from squidasm.util import get_qubit_state



class ServerProgram(Program):
    PEER = "client"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="server_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=8,
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
        fidelity = []

        # Receive EPR Pairs
        for i in range(num_qubits):
            if i == 0:
                brickwork[0] = recv_remote_state_preparation(epr_socket)
            else:
                brickwork.append(recv_remote_state_preparation(epr_socket))
        
        # reverse to match the qubit_id to place in list
        yield from conn.flush()
        brickwork.reverse()
        yield from conn.flush()

        # Prepare brickwork state from received qubits
        #"""
        brickwork[0].cphase(brickwork[1])
        brickwork[0].cphase(brickwork[3])
        brickwork[1].cphase(brickwork[2])
        brickwork[2].cphase(brickwork[5])
        brickwork[3].cphase(brickwork[4])
        brickwork[4].cphase(brickwork[7])
        brickwork[5].cphase(brickwork[6])
        brickwork[6].cphase(brickwork[7])
        #"""
    
        """
        for edge in G.edges:
            k, l = edge
            brickwork[k-1].cphase(brickwork[l-1])
        """
        yield from conn.flush()


        # Step 3: Measure and compute
        for i in range(num_qubits):
            # listen on csocket, progress if delta received
            delta = yield from csocket.recv_float()
            msg = yield from csocket.recv()
            assert msg == "delta sent"
            csocket.send("delta recieved")

            brickwork[i].rot_Z(angle=delta)
            brickwork[i].H()

            if i >= num_qubits-2:
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

        #print(f"fidelity: {fidelity}") 
        #print(f"number of measured qubits: {len(measurement)}")
        #print("m: ", measurement)

        return {"measurement": measurement, "m7": measurement[6], "m8": measurement[7], "f7": fidelity[0], "f8": fidelity[1]}
            