import numpy as np
from sklearn.cross_validation import KFold

from MultitaskDescriptor.mulas import MSE
from lasso_wrappers import DEFAULT_OPTIMIZER
from mumulader import MuMuLaDer
import itertools
from copy import deepcopy
__author__ = 'Victor'


class RandomizedMumulader(object):
    importance = None
    params = None
    model_class = MuMuLaDer

    def __init__(self,X, Y, tasks, descriptors, B, alpha,threshold, kwargs , loss_f=MSE):
        """
        lambda_1: regularization
        lambda_2: regularization
        B: Number of repetitions
        alpha: minim for sampling algorithm
        """
        self.X = X
        self.Y = Y
        self.tasks = tasks
        self.descriptors = descriptors
        self.B = B
        self.alpha = alpha
        self.kwargs = kwargs
        self.loss_f = loss_f
        self.threshold = threshold

    def optimize(self, optimizer=DEFAULT_OPTIMIZER, max_iters=100, max_inner_iters=10):
        n, p = self.X.shape
        _, k = self.tasks.shape
        model_intercept = self.kwargs.get('intercept', False)
        if self.kwargs.get('intercept',False):
            params = {
                'theta': np.zeros((1, p+1)),
                'selected':np.zeros((k,p+1))
            }
        else:
            params = {
                'theta': np.zeros((1,p)),
                'selected':np.zeros((k,p))
            }
        if self.kwargs.get('task_intercept',False):
            params['alpha'] = np.zeros((params['theta'].shape[1], self.descriptors.shape[1]+1))
        else:
            params['alpha'] = np.zeros((params['theta'].shape[1], self.descriptors.shape[1]))

        # First round of bootstrap
        for b in xrange(self.B):
            boots1 = np.random.choice(np.arange(n), size=n, replace=True)
            W = np.random.uniform(low= self.alpha, high=1.0, size=(1,p))
            new_X = self.X[boots1, :] * W
            new_Y = self.Y[boots1, :]

            new_tasks = self.tasks[boots1,:]
            model = self.model_class(new_X, new_Y, new_tasks, self.descriptors, **self.kwargs)
            model.optimize(optimizer=optimizer, max_iters=max_iters, max_inner_iters=max_inner_iters)
            beta = model.get_beta()
            params['selected'] += (beta != 0) / float(self.B)
            pars = model.get_params()
            params['theta'] += pars['theta'] / float(self.B)
            params['alpha'] += pars['alpha'] / float(self.B)
        # Second round of bootstrap


        self.params = params

    def get_params(self):
        return self.params

    def get_final_model(self):
        model = self.model_class(self.X,self.Y, self.tasks, self.descriptors, **self.kwargs)
        model.alpha = self.params['alpha']
        model.theta = self.params['theta']
        return model

###############################################################################
    def __cross_validation(self, values, std=False):
        lambda_1 = values['lambda_1']
        lambda_2 = values['lambda_2']
        B = values['B']
        threshold = values['threshold']
        alpha = values['alpha']

        mse = []
        for train, test in self.cv_indices:
            x_tr = self.X[train, :]
            y_tr = self.Y[train,:]
            x_te = self.X[test, :]
            y_te = self.Y[test,:]
            tasks_tr = self.tasks[train, :]
            tasks_te = self.tasks[test, :]
            descriptors = self.descriptors
            kwargs = deepcopy(self.kwargs)
            kwargs['lambda_1'] = lambda_1
            kwargs['lambda_2'] = lambda_2
            rmodel = RandomizedMumulader(x_tr, y_tr, tasks_tr, descriptors, B, threshold, alpha, kwargs)
            rmodel.optimize(max_iters=self.cross_max_iters)
            model = rmodel.get_final_model()
            y_pr = model.predict(x_te, tasks_te, self.descriptors)
            mse.append(self.loss_f(y_te.flatten(), y_pr.flatten()))
        if std:
            return np.mean(mse), np.std(mse)
        return np.mean(mse)

    def set_lambda_no_search(self, max_iters=100, n_folds=3, B_list=None, threshold_list=None, alpha_list=None, lambda_1_list=None, lambda_2_list=None):
        self.cv_indices = KFold(self.X.shape[0], n_folds=n_folds, shuffle=True)
        self.cross_max_iters = max_iters
        pars = []
        mean_scores = []
        std_scores = []
        for B, threshold, alpha, l_1, l_2 in itertools.product(B_list, threshold_list, alpha_list, lambda_1_list, lambda_2_list):
            params = {'B': B, 'threshold': threshold, 'alpha': alpha, 'lambda_1': l_1, 'lambda_2': l_2}
            #print params
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
                if par[key] < max_par[key]:
                    bigger = False
            if bigger:
                if np.abs(mean_scores[i] - max_mean) <= max_std:
                    valid_means.append(mean_scores[i])
                    valid_pars.append(par)
        ordered_indices = np.argsort(valid_means)
        best = valid_pars[ordered_indices[np.floor(len(ordered_indices)/2.)]]
        self.kwargs['lambda_1'] = best['lambda_1']
        self.kwargs['lambda_2'] = best['lambda_2']
        self.B = best['B']
        self.threshold = best['threshold']
        self.alpha = best['alpha']
        self.path = {'params': pars, 'scores': mean_scores, 'std_scores': std_scores}
        return best
