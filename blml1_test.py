#!/usr/bin/python3

import os

os.environ["NUMBA_BOUNDSCHECK"] = "1"
import random
import unittest

import numpy as np

import blml1


class Blml1Test(unittest.TestCase):
    def test_group_slices_of_contiguous_group_ids_v1(self):
        group_ids = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
        expected = [slice(0, 4), slice(4, 9), slice(9, 11)]
        actual = blml1.group_slices_of_contiguous_group_ids_v1(group_ids)
        self.assertEqual(expected, actual)

    def test_AccKOfV1_1(self):
        group_ids = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
        y_true = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        y_pred = [0, 1, 2, 3, 4, 4, 4, 4, 4, 1, 0]
        expected_count = (2, 3)
        expected_acc = expected_count[0] / expected_count[1]
        acc_k_of = blml1.AccKOfV1(
            3, blml1.group_slices_of_contiguous_group_ids_v1(group_ids)
        )
        self.assertEqual(expected_count, acc_k_of.count(y_true, y_pred))
        self.assertEqual(expected_acc, acc_k_of(y_true, y_pred))

    def test_split_n_by_rs_v1(self):
        for (n, rs), expected in (
            ((100, (1, 1)), [slice(0, 50), slice(50, 100)]),
            ((300, (1, 2)), [slice(0, 100), slice(100, 300)]),
            ((10, (1, 100)), [slice(0, 1), slice(1, 10)]),
            ((4, (1, 1, 1, 1)), [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4)]),
        ):
            actual = blml1.split_n_by_rs_v1(n, rs)
            self.assertEqual(expected, actual)

    def test_intersect_sorted_arrays_v1(self):
        for xss, expected in (
            (
                [
                    np.array([18, 22, 30]),
                    np.array([7, 11, 22, 24,]),
                    np.array([16, 22, 41]),
                ],
                [22],
            ),
            (
                [
                    np.array([25, 37, 39, 41, 46, 53, 56, 93, 95, 100,]),
                    np.array([34, 37, 43, 46, 59, 61, 95, 96,]),
                ],
                [37, 46, 95],
            ),
            (
                [
                    np.array([21, 60, 70, 84]),
                    np.array([27, 54, 67, 69, 77, 88, 93, 96]),
                ],
                [],
            ),
            ((), []),
            ([], []),
            ((np.array([1, 2]),), [1, 2]),
            ((np.array([1, 2]), np.array([3, 4])), []),
            ([np.array([1, 2]), np.array([3, 4])], []),
            ([np.array([1, 2]), np.array([2, 3, 4])], [2]),
            ([np.array([1]), np.array([1, 2])], [1]),
            ([np.array([1, 2]), np.array([1])], [1]),
            ([np.array([2]), np.array([1, 2])], [2]),
            ([np.array([1, 2]), np.array([2])], [2]),
            ([np.array([], dtype=int), np.array([1, 2])], []),
            ([np.array([1, 2]), np.array([], dtype=int)], []),
        ):
            self.assertEqual(list(blml1.intersect_sorted_arrays_v1(xss)), expected)
        rng = random.Random(42)
        for _ in range(400):
            xss = [
                np.array(
                    sorted(rng.randint(0, 100) for _ in range(rng.randint(0, 20))),
                    dtype=int,
                )
                for _ in range(rng.randint(0, 7))
            ]
            actual = list(blml1.intersect_sorted_arrays_v1(xss))
            expected = sorted(set.intersection(*map(set, xss)) if xss else [])
            self.assertEqual(
                actual, expected,
            )


if __name__ == "__main__":
    unittest.main()
