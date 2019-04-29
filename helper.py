import random


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
