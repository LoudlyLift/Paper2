import argparse
import ast
import collections
import math
import matplotlib.pyplot as plt
import numpy
import os
import random
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
parser.add_argument('--random-halflife', type=float, default=5000, help="The probability of choosing a random action is f(x)=1/x where x is the number of learning steps that have already been taken, except it has been scaled and shifted such that f(x)=1, and f(random-halflife)=0.5")
parser.add_argument('--learning-halflife', type=float, default=5000, help="The learning rate is f(x)=1/x where x is the number of learning steps that have already been taken, except it has been scaled and shifted such that f(x)=1, and f(learning-halflife)=0.5")
parser.add_argument('--future-discount', type=float, default=0.5, help="What weight to place on future values; zero ignores all future consequences, values near one give them significant weight (Denoted by γ in the paper)")

# MODEL
parser.add_argument('--num-servers', type=int, default=3, help="The number of edge devices that are available for the IoT device to offload tasks to (M)")
parser.add_argument('--num-parts', type=int, default=10, help="The number of parts that the task is split into (N_x)")

parser.add_argument('--offload-weight', type=float, default=1e-3, help="A weight that determines how much emphasis the optimization algorithm will place on offloading tasks")
parser.add_argument('--energy-weight', type=float, default=0.7, help="A weight that determines how important energy usage is to the optimization algorithm")
parser.add_argument('--latency-weight', type=float, default=1.0, help="A weight that determines how important latency is to the optimization algorithm")

parser.add_argument('--transmission-rates', type=Float2DMatrix, default=[[4e6, 5e6,  6e6,  7e6, 10e6],
                                                                         [8e6, 9e6, 10e6, 11e6, 12e6],
                                                                         [5e6, 6e6,  7e6,  9e6, 10e6]], help="A list. For each server, this list contains another list that contains the possible link rates with that server. There must be the same number of nested lists as there are servers, and each nested list must be the same length")
parser.add_argument('--transmission-transitions', type=Float2DMatrix, default=[[0.5,   0.5,   0,    0,     0],
                                                                               [0.1,   0.6,   0.3,  0,     0],
                                                                               [0,     0.2,   0.6,  0.2,   0],
                                                                               [0,     0,     0.3,  0.6,   0.1],
                                                                               [0,     0,     0,    0.5,   0.5]], help="A 2D matrix that defines how likely it is to transfer from any one transmission state to another")
parser.add_argument('--cycles-per-bit', type=int, default=1000, help="The number of CPU cycles it takes to process one bit of input data")
parser.add_argument('--effective-capacitance', type=float, default=1e-28, help="The effective capacitance coefficient of the CPU's chip architecture")
parser.add_argument('--clock-frequency', type=float, default=1e9, help="The device's CPU's max clock frequency")
parser.add_argument('--num-freqs', type=int, default=1, help="The number of discrete clock frequencies to run at")
parser.add_argument('--freq-delta-factor', type=float, default=0.3, help="Clock frequency i+1 = (1-delta_factor)*{clock frequency i}")

parser.add_argument('--transmit-power', type=float, default=0.5, help="The transmit power of the device itself (Watts)")

#TODO: find actual value
parser.add_argument('--data-gen-rate', type=float, default=120e3, help="The rate at which the device generates tasks/data for processing (C^(k); units of bits (?) per time interval)")

args = parser.parse_args()

if(len(args.transmission_rates) != args.num_servers):
    print("The length of the list passed to --transmission-rates must equal the value of --num-servers", file=sys.stderr)
    exit(1)
class environment(model.model):
    def __init__(self, args):
        super().__init__(cServers=args.num_servers, cParts=args.num_parts,
                         transmission_rates=args.transmission_rates,
                         data_gen_rate=args.data_gen_rate,
                         energy_weight=args.energy_weight,
                         latency_weight=args.latency_weight,
                         cycles_per_bit=args.cycles_per_bit,
                         effective_capacitance=args.effective_capacitance,
                         transmit_power=args.transmit_power,
                         transmission_transitions=args.transmission_transitions,
                         offload_weight=args.offload_weight)

        self.isTrainable = True

        servers = range(args.num_servers) # [0, num_servers)
        parts = range(args.num_parts+1)   # [0, num_parts]
        freqs = range(args.num_freqs)     # [0, num_freqs)
        self.possible_actions = list(itertools.product(servers, parts, freqs))

        self.move_legality = [True,] * len(self.possible_actions)

    def step(self, actionNum):
        (selection, nOffload, iFreq) = self.possible_actions[actionNum]
        freq = args.clock_frequency * (1-args.freq_delta_factor)**iFreq
        return super().step(selection, nOffload, freq)

    def getNumActions(self):
        return len(self.possible_actions)

    def getLegalMoves(self):
        return self.move_legality

    def randomAct(self):
        return random.randrange(self.getNumActions())

def preprocess(dicts, key, width):
    data = collections.deque(map(lambda dic: dic[key], dicts))
    data.reverse() #since we're using appendleft instead of right
    nSamples = len(data)

    assert(nSamples > width)
    if len(data) < 3*width:
        print("Warning: given width makes up a large fraction of the given data")

    tot = 0
    ys = []
    window = collections.deque()

    # initialize the right half of window
    cSamp = 0
    while cSamp < math.floor(width/2):
        sample = data.pop()
        tot += sample

        window.appendleft(sample)
        cSamp += 1

    # process data until the window is filled
    while cSamp < width:
        sample = data.pop()
        tot += sample
        ys.append(tot / cSamp)

        window.appendleft(sample)
        cSamp += 1

    # process data until we don't have any more samples to load into the window
    while len(data) > 0:
        sample = data.pop()

        tot -= window.pop()
        tot += sample
        window.appendleft(sample)
        #cSamp += 0

        ys.append(tot/cSamp)

    #import pdb; pdb.set_trace()
    # process data until the window is centered on the last sample
    for _ in range(math.ceil(width/2)):
        tot -= window.pop()
        cSamp -= 1

        ys.append(tot/cSamp)
    return ys

def plot(data, names, key, width=1000, ylabel=None, fName=None, fSuffix='.png'):
    """Plots a curve that is, in general, the centered moving average of the given
    width. When datapoints are missing, simply take the average of the available
    points.

    Data is an iterable of dictionaries. The values of those dictionaries
    corresponding to the key `key` are plotted and saved to the output
    directory.

    """
    assert(width >= 1)
    if ylabel is None:
        ylabel = key
    if fName is None:
        fName = ylabel

    data = [ preprocess(foo, key, width) for foo in data ]
    nSamples = len(data[0])

    sampleDensity = 100 #samples per pixel
    dpi = 300

    width = nSamples / sampleDensity / dpi
    maxWidth = 20
    #width = min(maxWidth, width)

    plt.figure(figsize=(width+2,6), dpi=dpi)
    plt.xlabel('Time Slot')
    plt.ylabel(ylabel)
    for (series, name) in zip(data, names):
        plt.plot(series, label=name)
    (ymin, ymax) = plt.ylim()
    plt.ylim(min(0, ymin), max(0, ymax))

    try:
        os.mkdir(args.dir_out)
    except FileExistsError as e:
        pass
    plt.legend(loc='upper left')
    plt.savefig(f'{args.dir_out}/{fName}{fSuffix}', bbox_inches='tight')
    plt.close()


def run(args):
    env = environment(args)
    player = qtable.qtable(env.getStateMetadata(), env.getNumActions(),
                           learning_rate_function=lambda step: max(0.01, args.learning_halflife / (step + args.learning_halflife)))

    ql = qlearning.qlearning(env=env, compute_randact=lambda step: max(0.02, args.random_halflife / (step + args.random_halflife)),
                             player=player, future_discount=args.future_discount)

    foo = ql.runEpisodes(training=True, episode_count=1, step_count=args.train_steps, log_episodes=0, log_steps=args.log_period)
    return foo[0] #because there's only one episode

args.num_freqs = 1
foo = run(args)
args.num_freqs = 9
bar = run(args)

for (key, ylabel, fName) in [("energyConsumption","Energy Consumption", "energy"),
                             ("latency", "Latency", "latency"),
                             ("utility","Utility", "utility")]:
    plot([foo,bar], ["Fixed Frequency", "Variable Frequency"], key, width=args.train_steps/25, ylabel=ylabel, fName=fName)

#results = ql.evaluate(args.eval_steps)
