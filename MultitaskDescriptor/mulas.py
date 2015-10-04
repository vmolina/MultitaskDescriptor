from __future__ import division

import matplotlib.pylab as plt

from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp

import numpy as np
import spams

def MSE(y_te, y_pred):
    return np.sqrt(mean_squared_error(y_te, y_pred))

class MuLas(object):
    beta = None
    def __init__(self, X, Y, tasks, lambda_1=None, beta=None, intercept=False,
                 name="multitaskLasso"):
        self.intercept = intercept
        self.name=name
        self.X = np.c_[np.ones(X.shape[0]), X] if self.intercept else X
        self.Y = Y
        self.tasks = tasks
        self.lambda_1 = 1.0 if lambda_1 is None else lambda_1
        self.num_data, self.output_dim, self.input_dim, self.tasks_dim = self.X.shape[0], self.Y.shape[1], self.X.shape[
            1], self.tasks.shape[1]

        if self.output_dim != 1:
            raise Exception("Y must be a (n,1) dimensional array")
        if beta is None:
            self.beta = np.zeros((self.tasks.shape[1], self.X.shape[1]),dtype=np.float64)
    def optimize(self, max_iters=None):
        p= self.X.shape[1]
        tasks_n = self.tasks.shape[1]
        X = np.asfortranarray(np.tile(self.X,(1,tasks_n))*np.repeat(self.tasks,p, axis=1))
        groups = np.asfortranarray(np.tile(np.arange(1,p+1).reshape(p),(tasks_n)), 'int32')
        beta0=np.asfortranarray(self.beta.reshape(p*tasks_n,1))
        self.beta= spams.fistaFlat(np.asfortranarray(self.Y), X, beta0, False, loss='square', regul='group-lasso-l2',lambda1=self.lambda_1, groups=groups).reshape(tasks_n,p)

    def predict(self, Xnew, tasks):
        X = np.c_[np.ones(Xnew.shape[0]), Xnew] if self.intercept else Xnew
        return np.sum(X * np.dot(tasks, self.beta), axis=1)

    def plot(self, resolution=20):
        if self.input_dim == 1:
            plt.scatter(self.X[:, int(self.intercept)], self.Y, color='k', marker='x')
            mi, ma = self.X.min(), self.X.max()
            ten_perc = .1 * (ma - mi)
            Xpred = np.repeat(np.linspace(mi - ten_perc, ma + ten_perc, resolution)[:, None], self.tasks_dim, axis=0)

            tasks = np.zeros((Xpred.shape[0], self.tasks_dim))
            for i in range(Xpred.shape[0]):
                tasks[i, i % self.tasks_dim] = 1

            mu = self.predict(Xpred, tasks)
            for i in range(self.tasks_dim):
                plt.plot(Xpred[tasks[:, i] == 1, :], mu[tasks[:, i] == 1], color='g', lw=1.5)
                # plt.fill_between(Xpred[:, 0], mu[:,0]+2*np.sqrt(np.diagonal(var)), mu[:,0]-2*np.sqrt(np.diagonal(var)), color='k', lambda_1=.1)
        else:
            raise NotImplementedError("Only one dim plots allowed")

    def get_params(self):
        return {
            'beta': self.beta,
        }

    def __cross_validation(self, values,std=False):
        lambda_1 = values['lambda_1']

        mse = []
        for train, test in self.cv_indices:
            x_tr = self.X[train, :]
            y_tr = self.Y[train]
            x_te = self.X[test, :]
            y_te = self.Y[test]
            tasks_tr = self.tasks[train, :]
            tasks_te = self.tasks[test, :]
            model = MuLas(x_tr, y_tr, tasks_tr, lambda_1=lambda_1,
                                   intercept=False)
            model.optimize(max_iters=self.cross_max_iters)
            y_pr = model.predict(x_te, tasks_te)
            mse.append(MSE(y_te, y_pr))
        if std:
            return np.mean(mse), np.std(mse)
        return np.mean(mse)

    def set_lambda(self, max_evals=10, max_iters=100, n_folds=3, max_lambda=10):
        self.cv_indices = KFold(self.X.shape[0], n_folds=n_folds, shuffle=True)
        self.cross_max_iters = max_iters
        space = hp.choice('model', [{
                                        'lambda_1': hp.uniform('lambda_1', 0, max_lambda)
                                    }])

        best = fmin(self.__cross_validation,
                    space=space, algo=tpe.suggest,
                    max_evals=max_evals
                    )
        self.lambda_1 = best['lambda_1']
    def set_lambda_no_search(self, max_iters=100, n_folds=3, lambda_1_list=None):
        self.cv_indices = KFold(self.X.shape[0], n_folds=n_folds, shuffle=True)
        self.cross_max_iters = max_iters
        pars = []
        mean_scores = []
        std_scores = []
        for l_1 in lambda_1_list:
            params = {'lambda_1':l_1}
            m_score, std_score = self.__cross_validation(params, std=True)
            mean_scores.append(m_score)
            std_scores.append(std_score)
            pars.append(params)
        max_i = np.argmin(mean_scores)
        max_par = pars[max_i]
        max_std = std_scores[max_i]
        max_mean = mean_scores[max_i]
        valid_pars = []
        valid_means = []
        for i, par in enumerate(pars):
            bigger = True
            for key in par:
                if  par[key] < max_par[key]:
                    bigger = False
            if bigger:
                if np.abs(mean_scores[i] - max_mean) <= max_std:
                    valid_means.append(mean_scores[i])
                    valid_pars.append(par)
        ordered_indices = np.argsort(valid_means)
        best = valid_pars[ordered_indices[np.floor(len(ordered_indices)/2.)]]
        self.lambda_1 = best['lambda_1']
        self.path = {'params':pars, 'scores':mean_scores, 'std_scores': std_scores}
        return best
    def get_lambda_1(self):
        return {'lambda_1': self.lambda_1}

    def init_params(self):
        self.beta = np.zeros((self.tasks_dim, self.X.shape[1]))
