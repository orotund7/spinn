from functools import partial
import argparse
import itertools

import numpy as np
from spinn import util

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

from chainer.functions.activation import slstm
from chainer.utils import type_check

from spinn.util.chainer_blocks import BaseSentencePairTrainer, Reduce
from spinn.util.chainer_blocks import LSTMState, Embed
from spinn.util.chainer_blocks import MLP
from spinn.util.chainer_blocks import CrossEntropyClassifier
from spinn.util.chainer_blocks import bundle, unbundle, the_gpu, to_cpu, to_gpu, treelstm


def HeKaimingInit(shape, real_shape=None):
    # Calculate fan-in / fan-out using real shape if given as override
    fan = real_shape or shape

    return np.random.normal(scale=np.sqrt(4.0/(fan[0] + fan[1])),
                            size=shape)


class SentencePairTrainer(BaseSentencePairTrainer):
    def init_params(self, **kwargs):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            if len(data.shape) >= 2:
                data[:] = HeKaimingInit(data.shape)
            else:
                data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, lr=0.01, **kwargs):
        self.optimizer = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
        # self.optimizer = optimizers.SGD(lr=0.01)
        self.optimizer.setup(self.model)
        # self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))
        # self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.00003))


class SentenceTrainer(SentencePairTrainer):
    pass


class BaseModel(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 input_keep_rate, classifier_keep_rate,
                 use_tracker_dropout=True, tracker_dropout_rate=0.1,
                 use_input_dropout=False, use_input_norm=False,
                 use_classifier_norm=True,
                 gpu=-1,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 use_tracking_lstm=True,
                 use_shift_composition=True,
                 use_history=False,
                 save_stack=False,
                 use_reinforce=False,
                 projection_dim=None,
                 encoding_dim=None,
                 use_encode=False,
                 use_skips=False,
                 use_sentence_pair=False,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        the_gpu.gpu = gpu

        mlp_input_dim = word_embedding_dim * 2 if use_sentence_pair else word_embedding_dim
        self.add_link('l0', L.Linear(mlp_input_dim, mlp_dim))
        self.add_link('l1', L.Linear(mlp_dim, mlp_dim))
        self.add_link('l2', L.Linear(mlp_dim, num_classes))

        self.classifier = CrossEntropyClassifier(gpu)
        self.__gpu = gpu
        self.accFun = accuracy.accuracy
        self.initial_embeddings = initial_embeddings

        vocab_size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        self.fix_embed = initial_embeddings is not None

        if initial_embeddings is None:
            self.add_link('_embed', L.EmbedID(vocab_size, word_embedding_dim))

    def embed(self, x, train):
        if self.fix_embed:
            return Variable(self.initial_embeddings.take(x, axis=0), volatile=not train)
        else:
            return self._embed(Variable(x, volatile=not train))

    def run_mlp(self, h, train):
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = h
        return h


class SentencePairModel(BaseModel):
    def __call__(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]

        embeds_prem = self.embed(x_prem, train)
        embeds_hyp = self.embed(x_hyp, train)

        h_prem = F.sum(embeds_prem, axis=1)
        h_hyp = F.sum(embeds_hyp, axis=1)
        h = F.concat([h_prem, h_hyp], axis=1)
        y = self.run_mlp(h, train)

        # Calculate Loss & Accuracy.
        accum_loss = self.classifier(y, Variable(y_batch, volatile=not train), train)
        self.accuracy = self.accFun(y, self.xp.array(y_batch))

        return y, accum_loss, self.accuracy.data, 0.0, None


class SentenceModel(BaseModel):
    def __call__(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        embeds = self.embed(x, train)

        h = F.sum(embeds, axis=1)
        y = self.run_mlp(h, train)

        # Calculate Loss & Accuracy.
        accum_loss = self.classifier(y, Variable(y_batch, volatile=not train), train)
        self.accuracy = self.accFun(y, self.xp.array(y_batch))

        return y, accum_loss, self.accuracy.data, 0.0, None
