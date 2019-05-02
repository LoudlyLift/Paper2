import typing
import numpy
import math

class qtable:
    """See qlearning.py for the definitions of state_metadata, state, and qValues
    """

    def __init__(self, state_metadata: typing.Tuple[int, ...], num_actions: int, learning_rate_function, initial_scale: float=0.001):
        self._table = numpy.random.random(state_metadata + (num_actions,)) * initial_scale - 10
        self._update_count = 0
        self._fLR = learning_rate_function

    def computeQState(self, state):
        return self._table[state].copy()

    def updateQState(self, state, targetQs):
        lr = self._fLR(self._update_count)
        currentQs = self.computeQState(state)

        self._table[state] = lr * targetQs + (1 - lr) * currentQs
        self._update_count += 1

    def getUpdateCount(self):
        return self._update_count
