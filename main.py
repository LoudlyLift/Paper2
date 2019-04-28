import argparse
import numpy
import time

import model
import qtable

assert(__name__ == "__main__")

parser = argparse.ArgumentParser(description='Implementation of the paper "Learning-Based Computation Offloading for IoT Devices With Energy Harvesting" by M. Min, et. al. Published in IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 68, NO. 2, FEBRUARY 2019')

# META
parser.add_argument('--log-period', type=int, default=100, help="How many steps to take between console updates (non-positive values disable logging)")

# Q-LEARNING
parser.add_argument('--num-steps', type=int, default=10000, help="How many steps to take")
parser.add_argument('--fraction-randomized', type=float, default=0.1, help="What fraction ([0,1]) of actions should be chosen at random")
parser.add_argument('--learning-rate', type=float, default=0.1, help="The Q-Table updates with a moving average. This is the weight of the most recent observation. Equivalent to 1-α from the paper.")

# MODEL
parser.add_argument('--num-servers', type=int, default=3, help="The number of edge devices that are available for the IoT device to offload tasks to (M)")
parser.add_argument('--num-parts', type=int, default=10, help="The number of parts that the task is split into (N_x)")

args = parser.parse_args()

class environment(model.model):
    def __init__(self, args):
        super().__init__(args.num_servers, args.num_parts)
        raise Exception("Not Implemented")

    def step(self, actionNum):
        #super().step(selection, nOffload)
        raise Exception("Not Implemented") #TODO: convert via lookup table?

    def getNumActions(self):
        raise Exception("Not Implemented")

    def getLegalMoves(self):
        raise Exception("Not Implemented")

env = environment(args)

ql = qlearning.qlearning(env=env, compute_randact=lambda: args.fraction_randomized,
                         consPlayer=qtable.qtable,
                         player_config={"learning_rate": args.learning_rate},
                         future_discount=args.future_discount)


print("Training Q-Table")
ql.train(args.train_episodes, log_period=args.log_period)

print("Evaluating Q-Table")
results = ql.evaluate(args.eval_episodes)
