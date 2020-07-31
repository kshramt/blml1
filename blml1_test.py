#!/usr/bin/python3

import os

os.environ["NUMBA_BOUNDSCHECK"] = "1"
import random
import unittest

import numpy as np

import blml1


class Blml1Test(unittest.TestCase):
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
            (([1, 2],), [1, 2]),
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
            self.assertEqual(blml1.intersect_sorted_arrays_v1(xss), expected)
        rng = random.Random(42)
        for _ in range(800):
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
