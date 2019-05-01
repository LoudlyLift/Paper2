import argparse
import ast
import matplotlib.pyplot as plt
import numpy
import statistics
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
parser.add_argument('--dir-out', type=str, default="./Out", help="The directory to save results to")

# Q-LEARNING
parser.add_argument('--train-steps', type=int, default=30000, help="How many steps to take for training")
parser.add_argument('--eval-steps', type=int, default=3000, help="How many steps to take during evaluation")
parser.add_argument('--fraction-randomized', type=float, default=0.1, help="What fraction ([0,1]) of actions should be chosen at random")
parser.add_argument('--learning-rate', type=float, default=0.1, help="The Q-Table updates with a moving average. This is the weight of the most recent observation. Equivalent to 1-α from the paper.")
parser.add_argument('--future-discount', type=float, default=0.5, help="What weight to place on future values; zero ignores all future consequences, values near one give them significant weight (Denoted by γ in the paper)")

# MODEL
parser.add_argument('--num-servers', type=int, default=3, help="The number of edge devices that are available for the IoT device to offload tasks to (M)")
parser.add_argument('--num-parts', type=int, default=10, help="The number of parts that the task is split into (N_x)")
parser.add_argument('--num-battery-levels', type=int, default=21, help="The number of levels to quantize the battery into")
parser.add_argument('--num-harvest-levels', type=int, default=6,  help="The number of levels to quantize the estimate of harvested energy into")

parser.add_argument('--offload-weight', type=float, default=1e-3, help="A weight that determines how much emphasis the optimization algorithm will place on offloading tasks")
parser.add_argument('--energy-weight', type=float, default=0.7, help="A weight that determines how important energy usage is to the optimization algorithm")
parser.add_argument('--latency-weight', type=float, default=1.0, help="A weight that determines how important latency is to the optimization algorithm")

parser.add_argument('--max-battery', type=float, default=(7.4*3600), help="The maximum charge the battery can hold (Wh)")

parser.add_argument('--transmission-rates', type=Float2DMatrix, default=[[4e6,5e6,6e6,7e6,10e6],
                                                                         [8e6,9e6,10e6,11e6,12e6],
                                                                         [5e6,6e6,7e6,9e6,10e6]], help="A list. For each server, this list contains another list that contains the possible link rates with that server. There must be the same number of nested lists as there are servers, and each nested list must be the same length")
parser.add_argument('--transmission-transitions', type=Float2DMatrix, default=[[0.5,   0.3,   0.15, 0.049, 0.001],
                                                                               [0.2,   0.4,   0.3,  0.08,  0.02],
                                                                               [0.05,  0.25,  0.4,  0.25,  0.05],
                                                                               [0.02,  0.08,  0.3,  0.4,   0.2],
                                                                               [0.001, 0.049, 0.15, 0.3,   0.5]], help="A 2D matrix that defines how likely it is to transfer from any one transmission state to another")
parser.add_argument('--cycles-per-bit', type=int, default=1000, help="The number of CPU cycles it takes to process one bit of input data")
parser.add_argument('--effective-capacitance', type=float, default=1e-28, help="The effective capacitance coefficient of the CPU's chip architecture")
parser.add_argument('--clock-frequency', type=float, default=1e9, help="The device's CPU's fixed clock frequency") #TODO: A, is clock frequency even supposed to be fixed? and B, what is the correct value?
parser.add_argument('--drop-penalty', type=float, default=10, help="The cost of letting the battery level reach zero (ψ in the paper)")

parser.add_argument('--transmit-power', type=float, default=0.5, help="The transmit power of the device itself (Watts)")

#TODO: find actual value
parser.add_argument('--data-gen-rate', type=float, default=120e3, help="The rate at which the device generates tasks/data for processing (C^(k); units of bits (?) per time interval)")

#TODO: calculate this dynamically
parser.add_argument('--max-harvest', type=float, default=(1.7e-3 * 3600), help="The maximum amount of energy that CAN be harvested in a single tick (only affects quantization levels? actual max depends on the other parameters)")

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
                         args.transmit_power, args.transmission_transitions,
                         args.offload_weight)

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

print("Transfer training")
ql.runEpisodes(training=True, episode_count=10000, step_count=20, log_episodes=100, log_steps=0)

print("Actual training")
foo = ql.runEpisodes(training=True, episode_count=1, step_count=args.train_steps, log_episodes=0, log_steps=args.log_period)
foo = foo[0] #because there's only one episode

def plot(dicts, key, weight=0.01, ylabel=None, fName=None, fSuffix='.png'):
    """Plots an exponential moving average. Data is an iterable of dictionaries. The
    values corresponding to the key `key` are plotted and saved to the output
    directory.

    """
    assert(0 < weight and weight <= 1)
    if ylabel is None:
        ylabel = key
    if fName is None:
        fName = ylabel

    data = list(map(lambda dic: dic[key], dicts))

    ys = []

    #The initial value for the exponential average is the average of the first
    #few points. We don't just want to initialize it with data[0] because that
    #gives that observation far more weight than it deserves.
    cInit = round(1/weight)
    val = statistics.mean(data[:cInit])

    for sample in data:
        val = weight*sample + (1-weight)*val
        ys.append(val)

    plt.xlabel('Time Slot')
    plt.ylabel(ylabel)
    plt.plot(ys)
    (ymin, ymax) = plt.ylim()
    plt.ylim(min(0, ymin), ymax)
    plt.savefig(f'{args.dir_out}/{fName}{fSuffix}', bbox_inches='tight')
    plt.close()

for (key, ylabel, fName) in [("energyConsumption","Energy Consumption", "1-energy"),
                             ("latency", "Latency", "2-latency"),
                             ("dropped", "Task Drop Rate", "3-drop"),
                             ("utility","Utility","4-utility")]:
    plot(foo, key, weight=0.001, ylabel=ylabel, fName=fName)

#results = ql.evaluate(args.eval_steps)
