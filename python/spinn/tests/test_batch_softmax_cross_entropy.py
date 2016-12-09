import unittest

import numpy as np

from spinn.util.batch_softmax_cross_entropy import batch_weighted_softmax_cross_entropy

# Chainer imports
import chainer
from chainer import reporter, initializers
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
from chainer.functions.connection import embed_id
from chainer.functions.normalization.batch_normalization import batch_normalization
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer.training import extensions


class BatchSoftmaxCrossEntropyTestCase(unittest.TestCase):

    def test_forward_uniform_w(self):

        X = np.array([
            [0.4, 0.6],
            [0.5, 0.5],
        ], dtype=np.float32)

        y = np.array([
            1,
            1,
        ], dtype=np.int32)

        w = np.array([
            0.5,
            0.5,
        ], dtype=np.float32)

        np.testing.assert_equal(
            F.softmax_cross_entropy(X, y).data,
            batch_weighted_softmax_cross_entropy(X, y, w).data
            )

    def test_forward_different_w(self):

        X = np.array([
            [0.4, 0.6],
            [0.5, 0.5],
        ], dtype=np.float32)

        y = np.array([
            1,
            1,
        ], dtype=np.int32)

        w = np.array([
            0.2,
            0.8,
        ], dtype=np.float32)

        xent = F.softmax_cross_entropy(X, y).data
        weighted_xent = batch_weighted_softmax_cross_entropy(X, y, w).data

        np.testing.assert_equal(np.any(np.not_equal(xent, weighted_xent)), True)
        np.testing.assert_almost_equal(xent, 0.645643055439)
        np.testing.assert_almost_equal(weighted_xent, 0.674145519733)


if __name__ == '__main__':
    unittest.main()
