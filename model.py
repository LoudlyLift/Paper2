import numpy

def model:
    """This class manages the state of the system in response to the device's
    actions, and evaluates the utility function.

    """

    def __init__(self, nServers, nParts):
        self.N_SERVERS = nServers
        self.N_PARTS = nParts
        raise Exception("Not Implemented")

    def reset(self):
        raise Exception("Not Implemented")

    def getStateShape(self):
        raise Exception("Not Implemented")

    def step(self, selection, nOffload):
        """Simulates a single timestep. The device elects to offload `nOffload`
        of the `N_PARTS` parts to the server indicated by the index `selection`.

        """
        assert(type(nOffload) == int) # don't judge
        assert(0 <= nOffload and nOffload <= self.N_PARTS) #yes, equality on both sides

        assert(type(selection) == int)
        assert(0 <= selection and selection < self.N_SERVERS)

        raise Exception("Not Implemented")

    def getResults(self):
        """Returns a list. That contains, for each time step, the tuple (energy
        consumption, computational latency, whether or not the task dropped,
        utility)

        """
        raise Exception("Not Implemented")
