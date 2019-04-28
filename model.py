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

        def statecount(self):
            return self.cStates

        def rateFromIndex(self, index):
            return self.rates[index]

    def __init__(self, cServers, cParts, cBatteryLevels, cHarvestLevels,
                 transmission_rates, max_battery, max_harvest):
        self.C_SERVERS = cServers
        self.C_PARTS = cParts
        self.C_BAT = cBatteryLevels
        self.C_HARVEST = cHarvestLevels
        self.MAX_BATTERY = max_battery
        self.MAX_HARVEST = max_harvest

        self.transmission_rates = transmission_rates

        self.reset()

    def getStateMetadata(self):
        return tuple(con.statecount() for con in self.connections) + (self.C_HARVEST, self.C_BAT)

    def getState(self):
        bat = self.battery * self.C_BAT / self.MAX_BATTERY
        bat = round(bat)

        harv = self.harvest_est * self.C_HARVEST / self.MAX_HARVEST
        harv = round(harv)

        return tuple(self.datarates) + (harv,bat)

    def reset(self):
        #TODO?
        self.connections = [ model.connection(rates) for rates in self.transmission_rates ]
        self.battery = 0

        self.step_computations()
        return self.getState()


    def step_computations(self):
        self.datarates = [ con.step() for con in self.connections ]
        self.harvest_est = 0 #TODO

    def step(self, selection, nOffload):
        """Simulates a single timestep. The device elects to offload `nOffload`
        of the `C_PARTS` parts to the server indicated by the index `selection`.

        """
        assert(type(nOffload) == int)
        assert(0 <= nOffload and nOffload <= self.C_PARTS) #equality on both

        assert(type(selection) == int)
        assert(0 <= selection and selection < self.C_SERVERS)

        self.step_computations()
        state = self.getState()

        raise Exception("Not Implemented")

        done = False
        return (state, reward, done)

    def closeEpisode(self):
        """Returns a list. That contains, for each time step, the tuple (energy
        consumption, computational latency, whether or not the task dropped,
        utility)

        """
        raise Exception("Not Implemented")
