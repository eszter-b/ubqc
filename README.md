## Application
Executes Grover's search algorithm using universal blind quantum computation based on the works of [Broadbent et al.][1] and of [Gustiani and DiVincenzo][2].

## Current state:
Runs a two qubit Grover's search on a fixed graph state with 100% success rate 
for parameters set in `config/gereric_config.yaml`.

### Measurements
The following measurements are implemented to test the application's success rates for given parameters:
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

[1]: https://arxiv.org/abs/0807.4154  "Broadbent et al"
[2]: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.062422  "Gustiani and DiVincenzo"
