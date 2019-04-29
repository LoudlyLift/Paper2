import argparse
import ast
import numpy
import sys
import time

import model
import qtable
import qlearning
import itertools

def Float2DMatrix(string):
    mat = ast.literal_eval(string)
    if type(mat) != list:
        raise argparse.ArgumentTypeError(f'"{string!r}" is not a Float2DMatrix')

    if(len(mat) == 0):
        return mat

    #width=len(mat[0])

    for row in mat:
        if type(row) != list:
            raise argparse.ArgumentTypeError(f'"{string!r}" is not a Float2DMatrix')
        #if len(row) != width:
        #    raise argparse.ArgumentTypeError(f'"{string!r}" is not a Float2DMatrix, as some rows are different widths')
        for entry in row:
            if type(entry) != float:
                raise argparse.ArgumentTypeError(f'"{string!r}" is not a Float2DMatrix')

    return mat

assert(__name__ == "__main__")

parser = argparse.ArgumentParser(description='Implementation of the paper "Learning-Based Computation Offloading for IoT Devices With Energy Harvesting" by M. Min, et. al. Published in IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 68, NO. 2, FEBRUARY 2019')

# META
parser.add_argument('--log-period', type=int, default=1000, help="How many steps to take between console updates (non-positive values disable logging)")

# Q-LEARNING
parser.add_argument('--train-steps', type=int, default=30000, help="How many steps to take for training")
parser.add_argument('--eval-steps', type=int, default=1000, help="How many steps to take during evaluation")
parser.add_argument('--fraction-randomized', type=float, default=0.1, help="What fraction ([0,1]) of actions should be chosen at random")
parser.add_argument('--learning-rate', type=float, default=0.1, help="The Q-Table updates with a moving average. This is the weight of the most recent observation. Equivalent to 1-α from the paper.")
parser.add_argument('--future-discount', type=float, default=0.5, help="What weight to place on future values; zero ignores all future consequences, values near one give them significant weight (Denoted by γ in the paper)")

# MODEL
parser.add_argument('--num-servers', type=int, default=3, help="The number of edge devices that are available for the IoT device to offload tasks to (M)")
parser.add_argument('--num-parts', type=int, default=10, help="The number of parts that the task is split into (N_x)")
parser.add_argument('--num-battery-levels', type=int, default=21, help="The number of levels to quantize the battery into")
parser.add_argument('--num-harvest-levels', type=int, default=6,  help="The number of levels to quantize the estimate of harvested energy into")

parser.add_argument('--energy-weight', type=float, default=0.7, help="A weight that determines how important energy usage is to the optimization algorithm")
parser.add_argument('--latency-weight', type=float, default=1.0, help="A weight that determines how important latency is to the optimization algorithm")

parser.add_argument('--max-battery', type=float, default=7.4, help="The maximum charge the battery can hold (Wh)")

parser.add_argument('--transmission-rates', type=Float2DMatrix, default=[[3,4,5,6,7],[8,9,10,11,12],[5,6,7,9,10]], help="A list. For each server, this list contains another list that contains the possible link rates with that server. There must be the same number of nested lists as there are servers, and each nested list must be the same length")
parser.add_argument('--cycles-per-bit', type=int, default=1000, help="The number of CPU cycles it takes to process one bit of input data")
parser.add_argument('--effective-capacitance', type=float, default=1e-11, help="The effective capacitance coefficient of the CPU's chip architecture")
parser.add_argument('--clock-frequency', type=float, default=1e9, help="The device's CPU's fixed clock frequency") #TODO: A, is clock frequency even supposed to be fixed? and B, what is the correct value?
parser.add_argument('--drop-penalty', type=float, default=10, help="The cost of letting the battery level reach zero (ψ in the paper)")

parser.add_argument('--transmit-power', type=float, default=0.5, help="The transmit power of the device itself (Watts)")

#TODO: find actual value
parser.add_argument('--data-gen-rate', type=float, default=(120e3*8), help="The rate at which the device generates tasks/data for processing (C^(k); units of bits (?) per time interval)")

#TODO: calculate this dynamically
parser.add_argument('--max-harvest', type=float, default=1.7e-3, help="The maximum amount of energy that CAN be harvested in a single tick (only affects quantization levels? actual max depends on the other parameters)")

args = parser.parse_args()

if(len(args.transmission_rates) != args.num_servers):
    print("The length of the list passed to --transmission-rates must equal the value of --num-servers", file=sys.stderr)
    exit(1)

class environment(model.model):
    def __init__(self, args):
        super().__init__(args.num_servers, args.num_parts,
                         args.num_battery_levels, args.num_harvest_levels,
                         args.transmission_rates, args.max_battery,
                         args.max_harvest, args.data_gen_rate,
                         args.energy_weight, args.latency_weight,
                         args.drop_penalty, args.cycles_per_bit,
                         args.effective_capacitance, args.clock_frequency,
                         args.transmit_power)

        self.isTrainable = True

        servers = range(args.num_servers) # [0, num_servers)
        parts = range(args.num_parts+1)   # [0, num_parts]
        self.possible_actions = list(itertools.product(servers, parts))

        self.move_legality = [True,] * len(self.possible_actions)

    def step(self, actionNum):
        (selection, nOffload) = self.possible_actions[actionNum]
        return super().step(selection, nOffload)

    def getNumActions(self):
        return len(self.possible_actions)

    def getLegalMoves(self):
        return self.move_legality

env = environment(args)

ql = qlearning.qlearning(env=env, compute_randact=lambda _: args.fraction_randomized,
                         consPlayer=qtable.qtable,
                         player_config={"learning_rate": args.learning_rate},
                         future_discount=args.future_discount)



print("Training Q-Table")
foo = ql.train(episode_count=1, step_count=args.train_steps, log_episodes=0, log_steps=args.log_period)
print(foo)

print("Evaluating Q-Table")
results = ql.evaluate(args.eval_steps)
