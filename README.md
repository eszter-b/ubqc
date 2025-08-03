***

# Universal Blind Quantum Computing over a Simulated Quantum Network

This project implements and simulates a Universal Blind Quantum Computing (UBQC) protocol over a two-node quantum network based on the works of [Broadbent et al.][1] and of [Gustiani and DiVincenzo][2].. The primary goal is to enable a client with limited quantum capabilities to securely delegate a quantum computation to a powerful, but untrusted, quantum server.
***
## Key Concepts

### Universal Blind Quantum Computing (UBQC)

UBQC is a cryptographic protocol that allows a client (who may only be able to prepare and send single qubits) to run a quantum algorithm on a remote quantum server without revealing their input, the computation being performed, or the output. This provides a pathway for a secure quantum cloud, where users can leverage the power of quantum computers without compromising their privacy.

The "blind" nature of the computation is achieved by the client sending qubits in a state that is randomized in a way only they know. The client then sends classical instructions to the server on how to measure these qubits. The server performs the measurements and returns the classical outcomes to the client, who can then correct for the initial randomization to obtain the final result of the computation.

### Measurement-Based Quantum Computing (MBQC)

This project is built upon the Measurement-Based Quantum Computing (MBQC) model. Unlike the more common circuit-based model where sequences of quantum gates are applied to qubits, MBQC performs computations through a series of single-qubit measurements on a highly entangled initial state, known as a cluster state or graph state.

The sequence of measurements and the choice of measurement bases determine the computation being performed. The outcomes of earlier measurements can influence the bases of later measurements, allowing for universal quantum computation.

## How It Works

The simulation involves a two-node network consisting of a **Client** and a **Server**:

1.  **Client-Side Preparation:** The client has a quantum computation they wish to perform. They encode this computation into a series of EPR (entangled) states with specific measurement angles. To ensure blindness, these angles are "hidden" by the client using a secret key.

2.  **Quantum Communication:** The client sends one half of the prepared EPR states to the server over a simulated quantum channel.

3.  **Server-Side Entanglement and Measurement:** The server receives the qubits from the client and entangles them to create a graph state. Following the client's classical instructions, the server measures the qubits in the specified bases. The server then returns the classical measurement outcomes to the client.

4.  **Client-Side Correction:** The client uses the server's measurement outcomes and their secret key to correct for the initial randomization and determine the result of their quantum computation.

## Simulators Used

This project leverages two key open-source Python libraries for simulation:

*   [**SquidASM**](https://www.squidasm.org/): Developed by QuTech, SquidASM is a powerful Software Development Kit (SDK) for simulating quantum networks. It allows for the detailed simulation of quantum applications running on different nodes connected by quantum channels. In this project, it is used to simulate the two-node (client-server) quantum network.

*   [**Graphix**](https://graphix.readthedocs.io/en/latest/): Graphix is a library specifically designed for the compilation, optimization, and simulation of Measurement-Based Quantum Computing. It provides tools to translate quantum circuits into MBQC measurement patterns and can optimize these patterns to reduce the required quantum resources.

***
## Implemented Measurements
The simulated measurements are collected in the `test_measurement.py` file.
The following measurements are implemented to test the application's success rates for given parameters:
### Proof-of-Concept:
* Success rate for every tagged state of 2-qubit Grover's search on a fixed graph.
* Success rate of the Deutsch-Jozsa algorithm for a balanced function.
* Creation of arbitrary brickwork states with testing for maximally mixed state.

### QDevice:
* server side: 
  * single qubit depolarisation probability
  * two qubit depolarisation probability
  * energy relaxation (T1) time - (T2 is also swept as model has T1 > T2 constraint)
  * (T2 found to be relevant under critical T1 value, for higher values the correction scheme compensates the phase flips)
* client side:
  * single qubit depolarisation probability
  * (client only prepares and measures her half of the Bell pairs -> no measurement for two qubit gates are needed)
  * (T1: qubits not kept in memory for long enough to be relevant)
  * (T2: qubits not kept in memory for long enough to be relevant)

### Quantum Channel:
* effect of fidelity of a depolarising channel

## Future plans:
Implement measurements to examine the effects of:
* distance between server and client
* parameters of a heralded channel

Make the application platform independent as it only works on SquidASM's trapped-ion quantum device model (extend to nitrogen vacancy centre model).
Test the validity and scalability of the application for different algorithms and graph states.

***
## Getting Started

To get started with this project, you will need to have Python 3 installed.
1. **Install SquidASM:**
For information on the installation of SquidASM refer to https://github.com/QuTech-Delft/squidasm. Note that the simulator only supports Unix-based OS, hence for Windows users [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) is recommended.
2. **Clone the repository:**
    ```bash
    git clone https://github.com/eszter-b/ubqc.git
    cd ubqc
    ```

3. **Install graphix:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install graphix
    ```


[1]: https://arxiv.org/abs/0807.4154  "Broadbent et al"
[2]: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.062422  "Gustiani and DiVincenzo"
