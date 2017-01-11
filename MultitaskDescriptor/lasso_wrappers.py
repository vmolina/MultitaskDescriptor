__author__ = 'victor'
try:
    import spams
    SPAMS = True
except Exception:
    SPAMS = False
import numpy as np


LASSO = 'Lasso'
LASSOLARS = 'LassoLars'
LASSOSPAMS = 'spams'
DEFAULT_OPTIMIZER = LASSOSPAMS if SPAMS else LASSO


class SpamWrapperOptimizer(object):
    def __init__(self, positive=False, lambda_=1., loss='square', regul='l1',warm_start=True, intercept=False,params=None, random_init=True):
        self.lambda_ = lambda_
        self.coef_ = None
        self.positive = positive
        self.warm_start = warm_start
        self.params = {} if params is None else params
        self.random_init = random_init
        self.intercept = intercept
        self.loss = loss
        self.regul = regul

    def fit(self, x, y, iters=100):
        if self.coef_ is None or not self.warm_start:
            self.initialize(x.shape)
        self.coef_ = spams.fistaFlat(np.asfortranarray(y, dtype=np.double), np.asfortranarray(x, dtype=np.double),
                                     np.asfortranarray(self.coef_.reshape((self.coef_.size, 1)), dtype=np.double), max_it=iters, lambda1=self.lambda_,
                                     loss=self.loss, regul=self.regul, intercept=self.intercept, **self.params).flatten()
        return self

    def initialize(self, x_shape):
        if self.random_init:
            self.coef_ = np.random.normal(size=(x_shape[1]))
        else:
            self.coef_ = np.zeros((x_shape[1]), dtype=np.float)