"""From built-in optimizer classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
from six.moves import zip

from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces

from keras.optimizers import Optimizer

class SGDW(Optimizer):
    """Stochastic gradient descent optimizer with decoupled weight decay.

    Includes support for momentum, learning rate decay, Nesterov momentum,
    and warm restarts.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        weight_decay: float >= 0. Normalized weight decay.
        eta: float >= 0. The multiplier to schedule learning rate and weight decay.
        steps_per_cycle: int > 0. The number of training batches of a restart cycle.
        
    # References
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, weight_decay=0.025, 
                 eta=1.0, steps_per_cycle=1, **kwargs):
        super(SGDW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.eta = K.variable(eta, name='eta')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.steps_per_cycle = K.variable(steps_per_cycle, name='steps_per_cycle')
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        w_d = self.eta*self.weight_decay/K.sqrt(self.steps_per_cycle)

        lr = self.eta*self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g - w_d * p
            else:
                new_p = p + v - w_d * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'eta': float(K.get_value(self.eta)),
                  'steps_per_cycle': int(K.get_value(self.steps_per_cycle))}
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay.
    Default parameters follow those provided in the original Adam paper.
    
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Normalized weight decay.
        eta: float >= 0. The multiplier to schedule learning rate and weight decay.
        steps_per_cycle: int > 0. The number of training batches of a restart cycle.
        
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.025, 
                 eta=1.0, steps_per_cycle=1, **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.eta = K.variable(eta, name='eta')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.steps_per_cycle = K.variable(steps_per_cycle, name='steps_per_cycle')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        w_d = self.eta*self.weight_decay/K.sqrt(self.steps_per_cycle)

        lr = self.eta*self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - w_d * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'eta': float(K.get_value(self.eta)),
                  'steps_per_cycle': int(K.get_value(self.steps_per_cycle)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
