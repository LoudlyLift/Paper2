import os
import random
import shelve
import time

def getCachedVariable(dbfile, varName, constructor, depFNames=[]):
    #The key for modification time has a randomized suffix to avoid collisions
    mTimeKey = varName + "_modificationTime_16159_12115"

    with shelve.open(dbfile) as shelf:
        if not (varName in shelf and mTimeKey in shelf):
            shelf[mTimeKey] = time.time()
            shelf[varName] = constructor()

        mTime = shelf[mTimeKey]
        for fName in depFNames:
            if os.path.getmtime(fName) > mTime:
                print(f"WARNING: {fName} has been modified since {varName} was computed")

        return shelf[varName]

def choose(iterable, filter=None):
    """Randomly select a single element from an arbitrary iterable that satisfies an
    arbitrary conditions. All of the elements that satisfy the condition are
    equally likely to be chosen.

    This function uses Ο(1) memory, and takes Ο(L) time, where L is the total
    number of items in the iterable. The iterable does not need to be
    repeatable, as it is only iterated through once. For all e in iterable, the
    call filter(e) is made exactly once.

    Using filter=None (the default) is equivalent to filter=lambda _: True

    e.g. given [0..9], choose a random odd number:

    helper.choose(range(10), lambda x: x%2 == 1)

    This can be verified to be correct easily:

    numpy.unique([ helper.choose(range(10), lambda x: x%2 == 1) for i in range(100000) ], return_counts=True)

    """
    randElement = None
    count = 0
    for element in iterable:
        if (filter is None or filter(element)):
            count += 1
            if (not random.randrange(count)):
                randElement = element

    return randElement
