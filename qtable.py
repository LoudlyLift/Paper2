import typing
import numpy
import math

class qtable:
    """See qlearning.py for the definitions of state_metadata, state, and qValues
    """

    def __init__(self, state_metadata: typing.Tuple[int, ...], num_actions: int, config=None):
        self._table = -1 * numpy.ones(state_metadata + (num_actions,))
        self._learning_rate = config['learning_rate']
        self._update_count = 0

    def computeQState(self, state):
        return self._table[state]

    def updateQState(self, _, state, qValues):
        self._update_count += 1

        lr = self._learning_rate
        val = self._table[state]

        val = lr * val + (1 - lr) * qValues

        self._table[state] = val

    def getUpdateCount(self):
        return self._update_count
