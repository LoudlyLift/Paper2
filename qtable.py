import typing
import numpy
import math

class qtable:
    """See qlearning.py for the definitions of state_metadata, state, and qValues
    """

    def __init__(self, state_metadata: typing.Tuple[int, ...], num_actions: int, learning_rate: float):
        self._table = -1 * numpy.ones(state_metadata + (num_actions,))
        self._learning_rate = learning_rate
        self._update_count = 0

    def computeQState(self, state):
        return self._table[state].copy()

    def updateQState(self, state, targetQs):
        lr = self._learning_rate
        currentQs = self.computeQState(state)

        self._table[state] = lr * targetQs + (1 - lr) * currentQs
        self._update_count += 1

    def getUpdateCount(self):
        return self._update_count
