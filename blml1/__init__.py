from typing import (
    Any,
    Dict,
    Final,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Sequence,
    TypeVar,
)
import contextlib
import random
import time

import lightgbm as lgb
import numba
import numpy as np
import optuna.integration.lightgbm

from ._common import logger


__version__ = "0.9.1"
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


@contextlib.contextmanager
def timing_v1(msg, fn=logger.info):
    t1 = time.monotonic()
    yield
    t2 = time.monotonic()
    fn(msg, t2 - t1)


def train_lightgbm_v1(
    data_train: lgb.Dataset,
    data_val: lgb.Dataset,
    params: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    params_hpo: Mapping[str, Any],
    kwargs_hpo: Mapping[str, Any],
) -> Dict[str, Any]:
    params_best: Dict[str, Any] = dict()
    tuning_history: List[Any] = []
    with timing_v1("Run optuna.integration.lightgbm.train: %s", logger.debug):
        model_hpo = optuna.integration.lightgbm.train(
            params_hpo,
            data_train,
            valid_sets=data_val,
            best_params=params_best,
            tuning_history=tuning_history,
            **kwargs_hpo,
        )
    logger.debug("model_hpo.best_score %s", model_hpo.best_score)
    logger.debug("params_best %s", params_best)
    params_fine = {**params_best, **params}
    logger.debug("params_fine %s", params_fine)
    with timing_v1("Run lgb.train: %s", logger.debug):
        model_fine = lgb.train(params_fine, data_train, valid_sets=data_val, **kwargs)
    return dict(
        model=model_fine,
        params=params_fine,
        params_best=params_best,
        tuning_history=tuning_history,
    )


def intersect1d_v1(xss: Sequence[Sequence[_T1]], assume_unique=False) -> Sequence[_T1]:
    n_xss = len(xss)
    if n_xss <= 0:
        return []
    elif n_xss == 1:
        return xss[0]
    else:
        xss = sorted(xss, key=len)
        ret = np.intersect1d(xss[0], xss[1], assume_unique=assume_unique)
        for i in range(2, n_xss):
            ret = np.intersect1d(ret, xss[i], assume_unique=assume_unique)
        return ret


def batch_v1(xs: Iterable[_T1], n: int) -> Generator[List[_T1], None, None]:
    if n < 1:
        raise ValueError(f"`n` should be >= 1: {n}")
    range_n = range(n)
    it = iter(xs)
    while True:
        try:
            yield [next(it) for _ in range_n]
        except StopIteration:
            return


def intersect_sorted_arrays_v1(xss):
    n_xss = len(xss)
    if n_xss <= 0:
        return []
    elif n_xss == 1:
        return _uniq_sorted_array_v1(xss[0])
    else:
        if len(xss[0]) <= 0:
            return []
        else:
            return _intersect_sorted_arrays_v1(tuple(xss))


@numba.njit(nogil=True, cache=True)
def _intersect_sorted_arrays_v1(xss):
    ret = []
    ns = [len(xs) for xs in xss]
    n_xss = len(xss)
    i_xs_last = n_xss - 1
    inds = np.zeros(n_xss, dtype=np.int64)
    i_xs = 0
    if ns[i_xs] <= inds[i_xs]:
        return ret
    x_max = xss[i_xs][inds[i_xs]]
    while True:
        i_xs += 1
        inds[i_xs] = _skip_lt_v1(xss[i_xs], inds[i_xs], ns[i_xs], x_max)
        if ns[i_xs] <= inds[i_xs]:
            return ret
        x = xss[i_xs][inds[i_xs]]
        if x_max < x:
            x_max = x
            i_xs = -1
        else:
            if i_xs == i_xs_last:
                ret.append(x_max)
                i_xs = 0
                inds[i_xs] = _skip_le_v1(xss[i_xs], inds[i_xs], ns[i_xs], x_max)
                if ns[i_xs] <= inds[i_xs]:
                    return ret
                x_max = xss[i_xs][inds[i_xs]]


@numba.njit(nogil=True, cache=True)
def _skip_lt_v1(xs, i, n, x_max):
    while i < n:
        x = xs[i]
        if x_max <= x:
            break
        i += 1
    return i


@numba.njit(nogil=True, cache=True)
def _skip_le_v1(xs, i, n, x_max):
    while i < n:
        x = xs[i]
        if x_max < x:
            break
        i += 1
    return i


@numba.njit(nogil=True, cache=True)
def _uniq_sorted_array_v1(xs):
    ret = []
    n_xs = len(xs)
    if n_xs == 0:
        return ret
    x = xs[0]
    ret.append(x)
    seen = x
    for i in range(1, n_xs):
        x = xs[i]
        if x != seen:
            ret.append(x)
            seen = x
    return ret
