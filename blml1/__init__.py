from typing import (
    Any,
    Dict,
    Final,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)
import contextlib
import itertools
import random
import time

import lightgbm as lgb
import numba
import numpy as np
import optuna.integration.lightgbm

from ._common import logger


__version__ = "0.16.0"
_T1 = TypeVar("_T1")

_CachedCallableV1_CACHE_V1 = dict()


class DataIteratorV1(Generic[_T1]):
    def __init__(
        self,
        xs: Sequence[_T1],
        random_state: int,
        shuffle: bool = True,
        repeat: bool = True,
    ):
        if not xs:
            raise ValueError("`xs` should have at least one element.")
        self._xs = xs
        self._rng = random.Random(random_state)
        self._shuffle = shuffle
        self._repeat = repeat
        self._n = len(self._xs)
        self._i = -1
        self._inds = list(range(self._n))

    def __iter__(self):
        return self

    def __next__(self) -> _T1:
        self._i += 1
        if self._repeat:
            self._i %= self._n
        elif self._n <= self._i:
            raise StopIteration
        if self._shuffle and (self._i == 0):
            self._rng.shuffle(self._inds)
        return self._xs[self._inds[self._i]]


class DataIterableV1(Generic[_T1]):
    def __init__(
        self, xs: Sequence[_T1], random_state: int, shuffle: bool = True, repeat=True
    ):
        self._xs = xs
        self._random_state = random_state
        self._shuffle = shuffle
        self._repeat = repeat

    def __iter__(self) -> DataIteratorV1[_T1]:
        return DataIteratorV1(
            self._xs, self._random_state, shuffle=self._shuffle, repeat=self._repeat
        )


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


class CachedCallableV1:
    """
    Example:

        import multiprocessing
        import functools


        def f(x, y):
            return x + y


        with multiprocessing.Pool(initializer=CachedCallableV1.set, initargs=("fn", functools.partial(f, 1))) as pool:
            ys = pool.map(CachedCallableV1("fn"), [1, 2, 3, 4])
    """

    __slots__ = ("_k",)

    @staticmethod
    def set(k, v):
        _CachedCallableV1_CACHE_V1[k] = v

    @staticmethod
    def delete(k):
        del _CachedCallableV1_CACHE_V1[k]

    def __init__(self, k):
        self._k = k

    def __call__(self, *args, **kwargs):
        return _CachedCallableV1_CACHE_V1[self._k](*args, **kwargs)


class AccKOfV1:
    def __init__(self, k, inds_list):
        """
        Parameters:
            group_ids: `group_ids` should be contiguous.

        Example:

            AccKOfV1(3, group_slices_of_contiguous_group_ids_v1([0, 0, 1, 1, 2, 2, 3, 4, 4, 4]))(y_true, y_pred)
        """
        self._k = k
        self._inds_list = inds_list

    def __call__(self, y_true, y_pred):
        n_ok, n_true = self.count(y_true, y_pred)
        return n_ok / n_true

    def count(self, y_true, y_pred):
        return (
            sum(
                in_top_k_v1(self._k, y_true[inds], y_pred[inds])
                for inds in self._inds_list
            ),
            len(self._inds_list),
        )


class LgbFevalV1:
    def __init__(self, fn, name: str, is_higher_better: bool, coef=1):
        self._fn = fn
        self._name = name
        self._is_higher_better = is_higher_better
        self._coef = coef

    def __call__(self, preds, dataset: lgb.Dataset):
        return (
            self._name,
            self._coef * self._fn(dataset.get_label(), preds),
            self._is_higher_better,
        )


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
    with timing_v1("Run optuna.integration.lightgbm.train: %s", logger.debug):
        model_hpo = optuna.integration.lightgbm.train(
            params_hpo,
            data_train,
            valid_sets=data_val,
            **kwargs_hpo,
        )
    params_best = model_hpo.params
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
        model_hpo=model_hpo,
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


def batch_v1(
    xs: Iterable[_T1], n: int, drop_reminder: bool = False
) -> Generator[List[_T1], None, None]:
    if n < 1:
        raise ValueError(f"`n` should be >= 1: {n}")
    it = iter(xs)
    while True:
        batch = list(itertools.islice(it, n))
        n_batch = len(batch)
        if (n_batch == 0) or (drop_reminder and (n_batch < n)):
            return
        yield batch


def intersect_sorted_arrays_v1(xss: Sequence[Sequence[_T1]]) -> Sequence[_T1]:
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


def split_n_by_rs_v1(n, rs):
    if any(r < 0 for r in rs):
        raise ValueError(f"any(r < 0 for r in rs): {rs}")
    total = sum(rs)
    if total <= 0:
        raise ValueError(f"total <= 0: {rs}")
    if n < len(rs):
        raise ValueError(f"n < len(rs): {n} {rs}")
    i2s = np.rint(np.cumsum(rs).astype(np.float64) * (n / total)).astype(np.int64)
    ret = []
    i1 = 0
    for i2 in i2s:
        if i2 <= i1:
            i2 = i1 + 1
        if n < i2:
            raise ValueError(f"n <= i2: {n} {rs} {i2s}")
        ret.append(slice(i1, i2))
        i1 = i2
    return ret


def in_top_k_v1(k, y_true, y_pred):
    """
    Parameters:
        k: This parameter is 1-origin.
    """
    n = len(y_pred)
    if n <= k:
        return True
    inds = np.where(np.equal(y_true, 1))[0]
    n_inds = len(inds)
    if n_inds <= 0:
        return False
    elif n_inds == 1:
        i = n - (k + 1)
        return np.partition(y_pred, i)[i] < y_pred[inds[0]]
    else:
        raise ValueError(f"y_true contains more than 1 positives: {y_true}")


def group_slices_of_contiguous_group_ids_v1(group_ids):
    if len(group_ids) <= 0:
        return []
    return list(_group_slices_of_contiguous_group_ids_v1(group_ids))


@numba.njit(nogil=True, cache=True)
def _intersect_sorted_arrays_v1(xss):
    ret = []
    ns = [len(xs) for xs in xss]
    n_xss = len(xss)
    i_xs_last = n_xss - 1
    inds = np.zeros(n_xss, dtype=np.int64)
    i_xs = 0
    if inds[i_xs] < ns[i_xs]:
        x_max = xss[i_xs][inds[i_xs]]
        while True:
            i_xs += 1
            inds[i_xs] = _skip_lt_v1(xss[i_xs], inds[i_xs], ns[i_xs], x_max)
            if ns[i_xs] <= inds[i_xs]:
                break
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
                        break
                    x_max = xss[i_xs][inds[i_xs]]
    return np.array(ret)


@numba.njit(nogil=True, cache=True)
def _skip_lt_v1(xs, i, n, x_max):
    while i < n:
        if x_max <= xs[i]:
            break
        i += 1
    return i


@numba.njit(nogil=True, cache=True)
def _skip_le_v1(xs, i, n, x_max):
    while i < n:
        if x_max < xs[i]:
            break
        i += 1
    return i


@numba.njit(nogil=True, cache=True)
def _uniq_sorted_array_v1(xs):
    ret = []
    n_xs = len(xs)
    if 0 < n_xs:
        x = xs[0]
        ret.append(x)
        seen = x
        for i in range(1, n_xs):
            x = xs[i]
            if x != seen:
                ret.append(x)
                seen = x
    return np.array(ret)


def _group_slices_of_contiguous_group_ids_v1(group_ids):
    n = len(group_ids)
    group_id_prev = group_ids[0]
    i1 = 0
    i2 = 0
    while True:
        i2 += 1
        if n <= i2:
            break
        group_id = group_ids[i2]
        if group_id != group_id_prev:
            yield slice(i1, i2)
            group_id_prev = group_id
            i1 = i2
    yield slice(i1, i2)
