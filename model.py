import numpy

import markov

class model:
    """This class manages the state of the system in response to the device's
    actions, and evaluates the utility function.

    """

    class connection(markov.chain):
        def __init__(self, rates):
            self.rates = rates
            self.cStates = len(self.rates)
            mat = [ [ 1 / self.cStates for dest in range(self.cStates) ]
                    for src in range(self.cStates)]

            super().__init__(mat)

        def step(self):
            index = super().step()
            return self.rates(cStates)

        def statecount(self):
            return self.cStates

    def __init__(self, cServers, cParts, cBatteryLevels, cHarvestLevels,
                 transmission_rates):
        self.C_SERVERS = cServers
        self.C_PARTS = cParts
        self.C_BAT = cBatteryLevels
        self.C_HARVEST = cHarvestLevels

    def getState(self):
        raise Exception("Not Implemented")

    def reset(self):
        #TODO?
        self.connections = [ model.connection(rates) for rates in transmission_rates ]
        return self.getState()

    def getStateMetadata(self):
        return tuple(con.statecount() for con in self.connections) + (self.C_HARVEST, self.C_BAT)

    def step(self, selection, nOffload):
        """Simulates a single timestep. The device elects to offload `nOffload`
        of the `C_PARTS` parts to the server indicated by the index `selection`.

        """
        assert(type(nOffload) == int)
        assert(0 <= nOffload and nOffload <= self.C_PARTS) #equality on both

        assert(type(selection) == int)
        assert(0 <= selection and selection < self.C_SERVERS)

        raise Exception("Not Implemented")

        state = getState()

        done = False
        return (state, reward, done)

    def closeEpisode(self):
        """Returns a list. That contains, for each time step, the tuple (energy
        consumption, computational latency, whether or not the task dropped,
        utility)

        """
        raise Exception("Not Implemented")
