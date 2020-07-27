from typing import Final, Generic, Sequence, TypeVar
import random

from ._common import logger


__version__ = "0.4.0"
_T1 = TypeVar("_T1")


class DataIteratorV1(Generic[_T1]):
    def __init__(self, xs: Sequence[_T1], random_state: int):
        if not xs:
            raise ValueError("`xs` should have at least one element.")
        self._xs = xs
        self._rng = random.Random(random_state)
        self._n = len(self._xs)
        self._i = -1
        self._inds = list(range(self._n))

    def __iter__(self):
        return self

    def __next__(self) -> _T1:
        self._i = (self._i + 1) % self._n
        if self._i == 0:
            self._rng.shuffle(self._inds)
        return self._xs[self._inds[self._i]]


class DataIterableV1(Generic[_T1]):
    def __init__(self, xs: Sequence[_T1], random_state: int):
        self._xs = xs
        self._random_state = random_state

    def __iter__(self) -> DataIteratorV1[_T1]:
        return DataIteratorV1(self._xs, self._random_state)


class LabelEncoderV1(Generic[_T1]):
    UNK_INT: Final = 0

    def __init__(self, xs: Sequence[_T1], unk_label: _T1):
        self._unk_label: Final = unk_label
        self._label_of_int: Final = [self._unk_label] + sorted(set(xs))
        self._int_of_label: Final = {
            x: i for i, x in enumerate(self._label_of_int) if i != self.UNK_INT
        }
        self.n_classes: Final = len(self._label_of_int)

    def encode(self, x: _T1) -> int:
        return self._int_of_label.get(x, self.UNK_INT)

    def decode(self, i: int) -> _T1:
        return self._label_of_int[i]
