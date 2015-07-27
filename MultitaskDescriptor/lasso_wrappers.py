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
    def __init__(self, positive=False, lambda_=1., warm_start=True, params=None):
        self.lambda_ = lambda_
        self.coef_ = None
        self.positive = positive
        self.warm_start = warm_start
        self.params = {} if params is None else params

    def fit(self, x, y):
        self.coef_ = spams.lasso(np.asfortranarray(y, dtype=np.float), D=np.asfortranarray(x, dtype=np.float),
                                 pos=self.positive, lambda1=self.lambda_, **self.params).toarray().flatten()
        return self
