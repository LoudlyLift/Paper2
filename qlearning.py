import itertools
import math
import numpy

import helper

def bestLegalMove(qvals, legality):
    qvals = qvals * legality
    return numpy.argmax(qvals)

class qlearning:
    """env state must be a tuple; no lists, no matricies, etc. A simple tuple of
    integers, and all integers in the range [0, max) must be valid, where max is
    the corresponding value from getStateMetadata. Moves must be representing by
    the integers [0, getNumActions()], however.

    env must define these methods:

        reset(self): resets env for a new game (including the first), and
        returns the starting state.

        step(self, int): perform the specified move and return the tuple
            (state_new,reward,done).

        getStateMetadata(self): returns a tuple of the same length as the state
        vector. Each entry is an integer specifying the number of values that
        entry can take in the actual state vector.

        getNumActions(self): returns the number of actions that can be made at
        any given time.

        getLegalMoves(self): returns a list of length == getNumActions(), whose
        entries are False if the move is illegal, and true if
        legal. (Equivalently, could be a list of zero/one)

        closeEpisode(self): returns the result of the episode

    compute_randact(episode_num): given the episode number, this computes
    probability with which a random move should be made instead of action
    chosen. Zero indexed.

    cls_player must make an instance using
    cls_player(state_shape, num_actions). That instance must have these
    methods:

        computeQState(self, state): returns a list of the estimated value of
        taking each enumerated action. (i.e. the row of the QTable
        corresponding to state)

        updateQState(self, state, qValues): do the player's equivalent of
        updating state's row in the Q-Table to match it's new estimated
        values.

    """
    def __init__(self, env, compute_randact, player, future_discount=.99):
        self._env = env
        self._compute_randact = compute_randact
        self._future_discount = future_discount
        self.player = player


        if self._env.isTrainable:
            self._train_update_count = 0
            self._train_episode_count = 0
        else:
            self._train_update_count = math.nan
            self._train_episode_count = math.nan

    def evaluate(self, episode_count=1, step_count=0, log_episodes=1, log_steps=0):
        """runs count episodes without updating Q-values.

        returns a list of the things returned by closeEpisode

        """
        return self.runEpisodes(training=False, episode_count=episode_count, step_count=step_count, log_episodes=log_episodes, log_steps=log_steps)

    def train(self, episode_count=1, step_count=0, log_episodes=1, log_steps=0):
        """runs count episodes while updating Q-Values.

        returns a list of the things returned by closeEpisode

        """
        if not self._env.isTrainable:
            return []
        return self.runEpisodes(training=True, episode_count=episode_count, step_count=step_count, log_episodes=log_episodes, log_steps=log_steps)

    def runEpisodes(self, training, episode_count, step_count, log_episodes, log_steps):
        if training:
            assert(self._env.isTrainable)
        results = []
        episode_count = max(1, episode_count)

        oldMat = self.player._table.copy()

        for i in range(1, episode_count+1):
            cStep = 0
            if log_episodes > 0 and i % log_episodes == 0:
                print(f"\rEPISODE: {i} / {episode_count}", end="" if log_steps==0 else "\n")
            try:
                state_old = self._env.reset()
                reward_sum = 0
                done = False

                while not done:
                    if step_count > 0 and log_steps > 0 and cStep % log_steps == 0:
                        #casual test show that computing the norm of the delta has <1% impact on performance
                        delta = self.player._table - oldMat
                        norm = numpy.linalg.norm(delta)
                        print(f"\r    STEP: {cStep} / {step_count}; {norm:15.3f}", end="")
                        oldMat = self.player._table.copy()
                    if step_count != 0 and cStep == step_count:
                        break
                    allActQs = self.player.computeQState(state_old)
                    if training and numpy.random.rand(1) < self._compute_randact(self._train_update_count):
                        act = self._env.randomAct()
                    else:
                        legalMoves = self._env.getLegalMoves()
                        act = bestLegalMove(allActQs, legalMoves)

                    state_new,reward,done = self._env.step(act)

                    if training:
                        if done:
                            maxHypotheticalQ = 0
                        else:
                            qvals = self.player.computeQState(state_new)
                            legalMoves = self._env.getLegalMoves()
                            maxHypotheticalQ = bestLegalMove(qvals, legalMoves)
                        allActQs[act] = reward + self._future_discount * maxHypotheticalQ
                        self.player.updateQState(state_old, allActQs)
                        self._train_update_count += 1

                    reward_sum += reward
                    state_old = state_new
                    cStep += 1
                results.append(self._env.closeEpisode())
                if training:
                    self._train_episode_count += 1
            except KeyboardInterrupt as e:
                if episode_count == 1:
                    results.append(self._env.closeEpisode())
                print("Keyboard Interrupt")
                break
        print("", end="\n")
        return results

    def getTrainUpdateCount(self):
        return self._train_update_count

    def getTrainEpisodeCount(self):
        return self._train_episode_count
