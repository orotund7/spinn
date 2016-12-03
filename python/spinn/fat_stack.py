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

"""
Style Guide:

1. Each __call__() or forward() should be documented with its
   input and output types/dimensions.
2. Every ChainList/Chain/Link needs to have assigned a __gpu and __mod.
3. Each __call__() or forward() should have `train` as a parameter,
   and Variables need to be set to Volatile=True during evaluation.
4. Each __call__() or forward() should have an accompanying `check_type_forward`
   called along the lines of:

   ```
   in_data = tuple([x.data for x in [input_1, input_2]])
   in_types = type_check.get_types(in_data, 'in_types', False)
   self.check_type_forward(in_types)
   ```

   This is mimicing the behavior seen in Chainer Functions.
5. Each __call__() or forward() should have a chainer.Variable as input.
   There may be slight exceptions to this rule, since at a times
   especially in this model a list is preferred, but try to stick to
   this as close as possible. When avoiding this rule, consider setting
   a property rather than passing the variable. For instance:

   ```
   link.transitions = transitions
   loss = link(sentences)
   ```
6. Each link should be made to run on GPU and CPU.
7. Type checking should be disabled using an environment variable.

"""

T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1

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

class Tracker(Chain):

    def __init__(self, size, tracker_size, predict, use_tracker_dropout=True, tracker_dropout_rate=0.1, use_skips=False):
        super(Tracker, self).__init__(
            lateral=L.Linear(tracker_size, 4 * tracker_size),
            buf=L.Linear(size, 4 * tracker_size, nobias=True),
            stack1=L.Linear(size, 4 * tracker_size, nobias=True),
            stack2=L.Linear(size, 4 * tracker_size, nobias=True))
        if predict:
            self.add_link('transition', L.Linear(tracker_size, 3 if use_skips else 2))
        self.state_size = tracker_size
        self.tracker_dropout_rate = tracker_dropout_rate
        self.use_tracker_dropout = use_tracker_dropout
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, bufs, stacks):
        self.batch_size = len(bufs)
        zeros = Variable(np.zeros(bufs[0][0].shape, dtype=bufs[0][0].data.dtype),
                         volatile='auto')
        buf = bundle(buf[-1] for buf in bufs)
        stack1 = bundle(stack[-1] if len(stack) > 0 else zeros for stack in stacks)
        stack2 = bundle(stack[-2] if len(stack) > 1 else zeros for stack in stacks)

        lstm_in = self.buf(buf.h)
        lstm_in += self.stack1(stack1.h)
        lstm_in += self.stack2(stack2.h)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            self.c = Variable(
                self.xp.zeros((self.batch_size, self.state_size),
                              dtype=lstm_in.data.dtype),
                volatile='auto')

        if self.use_tracker_dropout:
            lstm_in = F.dropout(lstm_in, self.tracker_dropout_rate, train=lstm_in.volatile == False)

        self.c, self.h = F.lstm(self.c, lstm_in)
        if hasattr(self, 'transition'):
            return self.transition(self.h)
        return None

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h


class SPINN(Chain):

    def __init__(self, args, vocab, normalization=L.BatchNormalization,
                 attention=False, attn_fn=None, use_reinforce=True, use_skips=False):
        super(SPINN, self).__init__(
            embed=Embed(args.size, vocab.size, args.input_dropout_rate,
                        vectors=vocab.vectors, normalization=normalization,
                        use_input_dropout=args.use_input_dropout,
                        use_input_norm=args.use_input_norm,
                        ),
            reduce=Reduce(args.size, args.tracker_size, attention, attn_fn))
        if args.tracker_size is not None:
            self.add_link('tracker', Tracker(
                args.size, args.tracker_size,
                predict=args.transition_weight is not None,
                use_tracker_dropout=args.use_tracker_dropout,
                tracker_dropout_rate=args.tracker_dropout_rate, use_skips=use_skips))
        self.transition_weight = args.transition_weight
        self.use_history = args.use_history
        self.save_stack = args.save_stack
        self.use_reinforce = use_reinforce
        self.use_skips = use_skips
        choices = [T_SHIFT, T_REDUCE, T_SKIP] if use_skips else [T_SHIFT, T_REDUCE]
        self.choices = np.array(choices, dtype=np.int32)

    def reset_state(self):
        self.memories = []

    def __call__(self, example, attention=None, print_transitions=False,
                 use_internal_parser=False, validate_transitions=True):
        self.bufs = self.embed(example.tokens)
        self.stacks = [[] for buf in self.bufs]
        self.buffers_t = [0 for buf in self.bufs]
        # There are 2 * N - 1 transitons, so (|transitions| + 1) / 2 should equal N.
        self.buffers_n = [(len([t for t in ts if t != T_SKIP]) + 1) / 2 for ts in example.transitions]
        for stack, buf in zip(self.stacks, self.bufs):
            for ss in stack:
                if self.save_stack:
                    ss.buf = buf[:]
                    ss.stack = stack[:]
                    ss.tracking = None
        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if hasattr(example, 'transitions'):
            self.transitions = example.transitions
        self.attention = attention
        return self.run(run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def validate(self, transitions, preds, stacks, buffers_t, buffers_n):
        # TODO: Almost definitely these don't work as expected because of how
        # things are initialized and because of the SKIP action.

        DEFAULT_CHOICE = T_SHIFT
        cant_skip = np.array([p == T_SKIP and t != T_SKIP for t, p in zip(transitions, preds)])
        preds[cant_skip] = DEFAULT_CHOICE

        # Cannot reduce on too small a stack
        must_shift = np.array([len(stack) < 2 for stack in stacks])
        preds[must_shift] = T_SHIFT

        # Cannot shift if stack has to be reduced
        must_reduce = np.array([buf_t >= buf_n for buf_t, buf_n in zip(buffers_t, buffers_n)])
        preds[must_reduce] = T_REDUCE

        must_skip = np.array([t == T_SKIP for t in transitions])
        preds[must_skip] = T_SKIP

        return preds

    def run(self, print_transitions=False, run_internal_parser=False,
            use_internal_parser=False, validate_transitions=True):
        # how to use:
        # encoder.bufs = bufs, unbundled
        # encoder.stacks = stacks, unbundled
        # encoder.tracker.state = trackings, unbundled
        # encoder.transitions = ExampleList of Examples, padded with n
        # encoder.run()
        self.history = [[] for buf in self.bufs] if self.use_history is not None \
                        else itertools.repeat(None)

        transition_loss, transition_acc = 0, 0
        if hasattr(self, 'transitions'):
            num_transitions = self.transitions.shape[1]
        else:
            num_transitions = len(self.bufs[0]) * 2 - 3

        for i in range(num_transitions):
            if hasattr(self, 'transitions'):
                transitions = self.transitions[:, i]
                transition_arr = list(transitions)
            else:
            #     transition_arr = [0]*len(self.bufs)
                raise Exception('Running without transitions not implemented')

            cant_skip = np.array([t != T_SKIP for t in transitions])
            if hasattr(self, 'tracker') and (self.use_skips or sum(cant_skip) > 0):
                transition_hyp = self.tracker(self.bufs, self.stacks)
                if transition_hyp is not None and run_internal_parser:
                    transition_hyp = to_cpu(transition_hyp)
                    if hasattr(self, 'transitions'):
                        memory = {}
                        if self.use_reinforce:
                            probas = F.softmax(transition_hyp)
                            samples = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
                            samples[cant_skip] = [np.random.choice(self.choices, 1, p=proba)[0] for proba in probas.data[cant_skip]]

                            transition_preds = samples
                            hyp_acc = probas
                            hyp_xent = probas
                            truth_acc = transitions
                            truth_xent = samples
                        else:
                            transition_preds = transition_hyp.data.argmax(axis=1)
                            hyp_acc = transition_hyp
                            hyp_xent = transition_hyp
                            truth_acc = transitions
                            truth_xent = transitions

                        if validate_transitions:
                            transition_preds = self.validate(transition_arr, transition_preds,
                                self.stacks, self.buffers_t, self.buffers_n)

                        memory["logits"] = transition_hyp
                        memory["preds"]  = transition_preds

                        if not self.use_skips:
                            raise Exception("Skipping skips not implemented robustly.")

                            hyp_acc = hyp_acc.data[cant_skip]
                            truth_acc = truth_acc[cant_skip]

                            cant_skip_mask = np.tile(np.expand_dims(cant_skip, axis=1), (1, 2))
                            hyp_xent = F.split_axis(transition_hyp, transition_hyp.shape[0], axis=0)
                            hyp_xent = F.concat([hyp_xent[i] for i, y in enumerate(cant_skip) if y], axis=0)
                            truth_xent = truth_xent[cant_skip]

                            memory["preds_cm"] = np.array(transition_preds[cant_skip])
                            memory["truth_cm"] = np.array(transitions[cant_skip])
                        else:
                            memory["preds_cm"] = np.array(transition_preds[cant_skip])
                            memory["truth_cm"] = np.array(transitions[cant_skip])

                        memory["hyp_acc"] = hyp_acc
                        memory["truth_acc"] = truth_acc
                        memory["hyp_xent"] = hyp_xent
                        memory["truth_xent"] = truth_xent

                        transition_acc += F.accuracy(
                            hyp_acc, truth_acc.astype(np.int32))
                        transition_loss += F.softmax_cross_entropy(
                            hyp_xent, truth_xent.astype(np.int32),
                            normalize=False)

                        if use_internal_parser:
                            transition_arr = transition_preds.tolist()

                        self.memories.append(memory)

            lefts, rights, trackings, attentions = [], [], [], []
            batch = zip(transition_arr, self.bufs, self.stacks, self.history,
                        self.tracker.states if hasattr(self, 'tracker') and self.tracker.h is not None
                        else itertools.repeat(None),
                        self.attention if self.attention is not None
                        else itertools.repeat(None))

            for ii, (transition, buf, stack, history, tracking, attention) in enumerate(batch):
                must_shift = len(stack) < 2

                if transition == T_SHIFT: # shift
                    if self.save_stack:
                        buf[-1].buf = buf[:]
                        buf[-1].stack = stack[:]
                        buf[-1].tracking = tracking
                    stack.append(buf.pop())
                    self.buffers_t[ii] += 1
                    if self.use_history:
                        history.append(stack[-1])
                elif transition == T_REDUCE: # reduce
                    for lr in [rights, lefts]:
                        if len(stack) > 0:
                            lr.append(stack.pop())
                        else:
                            zeros = Variable(np.zeros(buf[0].shape,
                                dtype=buf[0].data.dtype),
                                volatile='auto')
                            if self.save_stack:
                                zeros.buf = buf[:]
                                zeros.stack = stack[:]
                                zeros.tracking = tracking
                            lr.append(zeros)
                    trackings.append(tracking)
                    attentions.append(attention)
                else:
                    if self.use_history:
                        history.append(buf[-1])  # pad history so it can be stacked/transposed
            if len(rights) > 0:
                reduced = iter(self.reduce(
                    lefts, rights, trackings, attentions))
                for transition, stack, history in zip(
                        transition_arr, self.stacks, self.history):
                    if transition == T_REDUCE: # reduce
                        new_stack_item = next(reduced)
                        assert isinstance(new_stack_item.data, np.ndarray), "Pushing cupy array to stack"
                        stack.append(new_stack_item)
                        if self.use_history:
                            history.append(stack[-1])
        if print_transitions:
            print()
        if self.transition_weight is not None:
            reporter.report({'transition_accuracy': transition_acc / num_transitions,
                             'transition_loss': transition_loss / num_transitions}, self)
            transition_loss *= self.transition_weight
        else:
            transition_loss = None

        return [stack[-1] for stack in self.stacks], transition_loss


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
                 use_skips=False,
                 use_sentence_pair=False,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        the_gpu.gpu = gpu

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim
        self.add_link('l0', L.Linear(mlp_input_dim, mlp_dim))
        self.add_link('l1', L.Linear(mlp_dim, mlp_dim))
        self.add_link('l2', L.Linear(mlp_dim, num_classes))

        if use_classifier_norm:
            self.add_link('bn_0', L.BatchNormalization(mlp_input_dim))
            self.add_link('bn_1', L.BatchNormalization(mlp_dim))
            self.add_link('bn_2', L.BatchNormalization(mlp_dim))

        self.classifier = CrossEntropyClassifier(gpu)
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.initial_embeddings = initial_embeddings
        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.use_classifier_norm = use_classifier_norm
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.use_reinforce = use_reinforce

        args = {
            'size': model_dim/2,
            'tracker_size': tracking_lstm_hidden_dim if use_tracking_lstm else None,
            'transition_weight': transition_weight,
            'use_history': use_history,
            'save_stack': save_stack,
            'input_dropout_rate': 1. - input_keep_rate,
            'use_input_dropout': use_input_dropout,
            'use_input_norm': use_input_norm,
            'use_tracker_dropout': use_tracker_dropout,
            'tracker_dropout_rate': tracker_dropout_rate,
        }
        args = argparse.Namespace(**args)

        vocab = {
            'size': initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size,
            'vectors': initial_embeddings,
        }
        vocab = argparse.Namespace(**vocab)

        self.add_link('spinn', SPINN(args, vocab, normalization=L.BatchNormalization,
                 attention=False, attn_fn=None, use_reinforce=use_reinforce, use_skips=use_skips))


    def build_example(self, sentences, transitions, train):
        raise Exception('Not implemented.')


    def run_spinn(self, example, train, use_internal_parser, validate_transitions=True):
        r = reporter.Reporter()
        r.add_observer('spinn', self.spinn)
        observation = {}
        with r.scope(observation):
            self.spinn.reset_state()
            h_both, _ = self.spinn(example,
                                   use_internal_parser=use_internal_parser,
                                   validate_transitions=validate_transitions)

        transition_acc = observation.get('spinn/transition_accuracy', 0.0)
        transition_loss = observation.get('spinn/transition_loss', None)
        return h_both, transition_acc, transition_loss


    def run_mlp(self, h, train):
        # Pass through MLP Classifier.
        h = to_gpu(h)
        if hasattr(self, 'bn_0'):
            h = self.bn_0(h, not train)
        h = F.dropout(h, self.classifier_dropout_rate, train)
        h = self.l0(h)
        h = F.relu(h)
        if hasattr(self, 'bn_1'):
            h = self.bn_1(h, not train)
        h = F.dropout(h, self.classifier_dropout_rate, train)
        h = self.l1(h)
        h = F.relu(h)
        if hasattr(self, 'bn_2'):
            h = self.bn_2(h, not train)
        h = F.dropout(h, self.classifier_dropout_rate, train)
        h = self.l2(h)
        y = h

        return y


    def __call__(self, sentences, transitions, y_batch=None, train=True,
                 use_internal_parser=False, validate_transitions=True):
        example = self.build_example(sentences, transitions, train)
        h, transition_acc, transition_loss = self.run_spinn(example, train, use_internal_parser, validate_transitions)
        y = self.run_mlp(h, train)

        # Calculate Loss & Accuracy.
        accum_loss = self.classifier(y, Variable(y_batch, volatile=not train), train)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

        if hasattr(transition_acc, 'data'):
          transition_acc = transition_acc.data

        return y, accum_loss, self.accuracy.data, transition_acc, transition_loss

class SentencePairModel(BaseModel):
    def build_example(self, sentences, transitions, train):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        assert batch_size * 2 == x.shape[0]
        assert batch_size * 2 == t.shape[0]

        example = {
            'tokens': Variable(x, volatile=not train),
            'transitions': t
        }
        example = argparse.Namespace(**example)

        return example


    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True):
        h_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions)
        batch_size = len(h_both) / 2
        h_premise = F.concat(h_both[:batch_size], axis=0)
        h_hypothesis = F.concat(h_both[batch_size:], axis=0)
        h = F.concat([h_premise, h_hypothesis], axis=1)
        return h, transition_acc, transition_loss


class SentenceModel(BaseModel):
    def build_example(self, sentences, transitions, train):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = {
            'tokens': Variable(x, volatile=not train),
            'transitions': t
        }
        example = argparse.Namespace(**example)

        return example


    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True):
        h, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions)
        h = F.concat(h, axis=0)
        return h, transition_acc, transition_loss
