import numpy
import math
import random

import markov

class model:
    """This class manages the state of the system in response to the device's
    actions, and evaluates the utility function.

    """

    class connection(markov.chain):
        def __init__(self, rates, transition):
            self.rates = rates
            self.cStates = len(self.rates)
            super().__init__(transition)

        def statecount(self):
            return self.cStates

        def rateFromIndex(self, index):
            return self.rates[index]

    def __init__(self, cServers, cParts, transmission_rates,
                 data_gen_rate, energy_weight, latency_weight,
                 cycles_per_bit, effective_capacitance, transmit_power,
                 transmission_transitions, offload_weight):
        #TODO: WAAY to many parameters here. Can argparse construct an object?
        #Then we could just pass that and save it. It'd make usage a little
        #uglier inside of this class (self.args.LATENCY_WEIGHT instead of
        #self.LATENCY_WEIGHT), but that'd still be an improvement over this,
        #I think.
        self.C_SERVERS = cServers
        self.C_PARTS = cParts
        self.DATA_GEN_RATE = data_gen_rate
        self.ENERGY_WEIGHT = energy_weight
        self.LATENCY_WEIGHT = latency_weight
        self.CYCLES_PER_BIT = cycles_per_bit
        self.TRANSMIT_POWER = transmit_power
        self.TRANSMISSION_RATES = transmission_rates
        self.EFFECTIVE_CAPACITANCE = effective_capacitance
        self.TRANSMISSION_TRANSITIONS = transmission_transitions
        self.OFFLOAD_WEIGHT = offload_weight

        self.reset()

    def getStateMetadata(self):
        return tuple(con.statecount() for con in self.connections)

    def getState(self):
        return tuple(self.iDataRates)

    def reset(self):
        self.connections = [ model.connection(rates, self.TRANSMISSION_TRANSITIONS) for rates in self.TRANSMISSION_RATES ]
        self.iDataRates = [ con.state for con in self.connections ]

        self.results = []

        return self.getState()

    def computation_step(self, selection, nOffload, freq):
        # Compute new connection rates
        self.iDataRates = [ con.step() for con in self.connections ]


        #x^{(k)}*C^{(k)}
        cOffloadBits = (nOffload/self.C_PARTS)*self.DATA_GEN_RATE

        #(1-x^{(k)}) * C^{(k)}
        cLocalBits = self.DATA_GEN_RATE - cOffloadBits

        #(1-x^{(k)}) * C^{(k)} * N
        cLocalCycles = cLocalBits * self.CYCLES_PER_BIT

        #B_i^{(k)}
        con = self.connections[selection]
        linkRate = con.rateFromIndex(self.iDataRates[selection])


        #(1)
        latencyLocal = cLocalCycles / freq

        #(3)
        latencyOffload = cOffloadBits / linkRate

        #(N/A): pg. 1934, left column, near middle at the end of a paragraph
        latency = max(latencyLocal, latencyOffload)


        #(2), but with fixed clock frequency
        energyLocal = cLocalCycles * self.EFFECTIVE_CAPACITANCE * \
            (freq**2)

        #(4)
        energyOffload = latencyOffload * self.TRANSMIT_POWER

        #(N/A): pg. 1933, directly above (5)
        energyConsumption = energyLocal + energyOffload

        tOffload = self.OFFLOAD_WEIGHT * cOffloadBits
        tEnergy = -self.ENERGY_WEIGHT * energyConsumption
        tLatency = -self.LATENCY_WEIGHT * latency

        # (8)
        utility  = tOffload + tEnergy + tLatency

        return { "utility": utility, "energyConsumption": energyConsumption,
                 "latency": latency, "fracOffload": nOffload/self.C_PARTS,
                 "freq": freq, }

    def step(self, selection, nOffload, freq):
        """Simulates a single timestep. The device elects to offload `nOffload`
        of the `C_PARTS` parts to the server indicated by the index `selection`.

        """
        assert(type(nOffload) == int)
        assert(0 <= nOffload and nOffload <= self.C_PARTS) #equality on both

        assert(type(selection) == int)
        assert(0 <= selection and selection < self.C_SERVERS)

        result = self.computation_step(selection, nOffload, freq)
        self.results.append(result)

        state = self.getState()

        done = False
        return (state, result["utility"], done)

    def closeEpisode(self):
        """Returns a list that contains, for each time step, a dictionary of useful
        values.

        """
        return self.results
