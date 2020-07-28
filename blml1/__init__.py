from typing import Any, Dict, Final, Generic, Mapping, Sequence, TypeVar
import contextlib
import random
import time

import lightgbm as lgb
import optuna.integration.lightgbm

from ._common import logger


__version__ = "0.5.0"
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
    params_best = dict()
    tuning_history = []
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
