import numpy
import random

class chain:
    def __init__(self, matrix, state = None):
        """sum(matrix[i]) == 1 for all i.
        matrix[i][j] is the probability of going from state i to state j

        """
        self.matrix = numpy.array(matrix).transpose()
        self.count = self.matrix.shape[0]
        assert(self.matrix.shape == (self.count, self.count))

        self.onehot = numpy.identity(self.count)

        if state is None:
            state = random.randrange(self.count)
        self.state = state

        self.indicies = list(range(self.count))

    def step(self):
        distribution = numpy.matmul(self.matrix, self.onehot[self.state])
        self.state = numpy.random.choice(self.indicies, p=distribution)
        return self.state
