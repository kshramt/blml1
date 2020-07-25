import random

from ._common import logger


__version__ = "0.1.0"


class DataIterableV1:
    def __init__(self, xs, random_state: int):
        self._xs = xs
        self._random_state = random_state

    def __iter__(self):
        return DataIteratorV1(self._xs, self._random_state)


class DataIteratorV1:
    def __init__(self, xs, random_state: int):
        self._xs = xs
        self._rng = random.Random(random_state)
        self._n = len(self._xs)
        self._i = -1
        self._inds = list(range(self._n))

    def __next__(self):
        self._i = (self._i + 1) % self._n
        if self._i == 0:
            self._rng.shuffle(self._inds)
        return self._xs[self._inds[self._i]]
