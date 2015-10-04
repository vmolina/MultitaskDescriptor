from __future__ import division

from sklearn.cross_validation import KFold
from sklearn.linear_model import Lasso, LassoLars

from hyperopt import fmin, tpe, hp

from mulas import MSE
from lasso_wrappers import *


class MuMuLas(object):
    theta = None
    gamma = None

    def __init__(self, x, y, tasks, intercept=False, lambda_1=1., lambda_2=1., gamma_params=None, theta_params=None,
                 tol=0.001, random_init=False):
        self.intercept = intercept
        self.x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=x.dtype)), axis=1) if self.intercept else x
        self.y = y if len(y.shape) == 2 else y.reshape((y.shape[0], 1))
        self.tasks = tasks

        self.lambda_1 = .1 if lambda_1 is None else lambda_1
        self.lambda_2 = .1 if lambda_2 is None else lambda_2

        self.theta = None
        self.gamma = None

        self.gamma_params = {} if gamma_params is None else gamma_params
        self.theta_params = {} if theta_params is None else theta_params
        self.tol = tol
        self.iterations = None
        self.random_init = random_init

    def optimize(self, optimizer=DEFAULT_OPTIMIZER, max_iters=100, ):
        """
        For fitting the model the following elements are required:
        x: A numpy array of shape (n,p) for the independent variables.
        y: A numpy array of haspe (n,1) for the dependent variables.
        tasks: A numpy array with (n,T) where every row contain just
            one element 1 and 0 for the other numbers to indicate to
            which task belongs.
        descriptors: An (T,L) array containing the different descriptors
            or each task.
        epsilon: A parameter for defining when to stop the optimization.
        """
        # get input sizes and initizialize parameters
        n, p = self.x.shape
        T = self.tasks.shape[1]

        # Precalculate x_gamma matrix for optimizing gamma
        # It is a (n, p*L) where xD_{i,(j-1)L+l) = \theta_{j}d_l for the
        # corresponding task.
        # Initialize theta
        self.theta = np.ones((1, p))

        if optimizer == LASSO:
            optimize_theta = Lasso(warm_start=True, positive=True, alpha=self.lambda_1, **self.theta_params)
            optimize_gamma = Lasso(warm_start=True, alpha=self.lambda_2, **self.gamma_params)
        elif optimizer == LASSOLARS:
            optimize_theta = LassoLars(alpha=self.lambda_1, **self.theta_params)
            optimize_gamma = LassoLars(alpha=self.lambda_2, **self.gamma_params)
        elif optimizer == LASSOSPAMS:
            optimize_theta = SpamWrapperOptimizer(positive=True, lambda_=self.lambda_1, params=self.theta_params)
            optimize_gamma = SpamWrapperOptimizer(lambda_=self.lambda_2, params=self.gamma_params)
        else:
            raise Exception("Not a valid value")
        for t in range(self.tasks.shape[1]):
            if self.random_init:
                self.gamma = np.random.normal(size=(T, p))
                #         optimize_gamma[t].coef_ = self.gamma[t,:].flatten()
            else:
                self.gamma = np.zeros((T, p), dtype=np.float64)
        # optimize_gamma[t].coef_ = self.gamma[t,:].flatten()

        beta_0 = self.get_beta()
        repeated_tasks = np.repeat(self.tasks, p, axis=0).reshape(n, p * T)

        # Start the two phase optimization
        continue_optimization = True
        self.iterations = 0
        while continue_optimization:

            # for t in range(self.tasks.shape[1]):
            #     x_t = np.dot(self.x[self.tasks[:, t].astype(bool), :], np.diag(self.theta.flatten()))
            #     y_t = self.y[self.tasks[:, t].astype(bool)]
            #     if len(x_t.flatten())> 0:
            #         self.gamma[t, :] = optimize_gamma[t].fit(x_t, y_t).coef_
            #
            # self.theta[:] =np.sqrt(self.lambda_2/self.lambda_1) *np.sqrt(np.sum(np.abs(self.get_beta()), axis=0))

            # Optimize for gamma
            gamma_x = np.repeat(self.theta * self.x, T, axis=1) * repeated_tasks
            self.gamma = optimize_gamma.fit(gamma_x,
                                            self.y).coef_.reshape(p, T).T

            # Optimize for theta
            x_theta = self.tasks.dot(self.gamma) * self.x
            self.theta = optimize_theta.fit(x_theta, self.y).coef_
            self.theta.shape = (1, len(self.theta))

            beta = self.get_beta()

            self.iterations += 1

            if np.linalg.norm(
                            self.get_beta().flatten() - beta_0.flatten()) < self.tol:  # np.linalg.norm(beta.flatten() - beta_0.flatten()) < self.tol:
                continue_optimization = False
            else:
                beta_0 = self.get_beta()
            if self.iterations >= max_iters:
                continue_optimization = False

    def get_beta(self):
        return self.theta * self.gamma

    def predict(self, x, tasks):
        new_x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=x.dtype)), axis=1) if self.intercept else x

        beta = self.get_beta()
        return np.sum(tasks.dot(beta) * new_x, axis=1)

    def get_params(self):
        return {
            'theta': self.theta,
            'gamma': self.gamma
        }

    def __cross_validation(self, values, std=False):
        lambda_1 = values['lambda_1']
        lambda_2 = values['lambda_2']

        mse = []
        for train, test in self.cv_indices:
            x_tr = self.x[train, :]
            y_tr = self.y[train, :]
            x_te = self.x[test, :]
            y_te = self.y[test, :]
            tasks_tr = self.tasks[train, :]
            tasks_te = self.tasks[test, :]

            model = MuMuLas(x_tr, y_tr, tasks_tr, lambda_1=lambda_1, lambda_2=lambda_2,
                            intercept=self.intercept, gamma_params=self.gamma_params,
                            theta_params=self.theta_params, tol=self.tol, random_init=self.random_init)
            model.optimize(max_iters=self.cross_max_iters)
            y_pr = model.predict(x_te, tasks_te)
            mse.append(MSE(y_te.flatten(), y_pr.flatten()))
            if std:
                return np.mean(mse), np.std(mse)
        return np.mean(mse)

    def set_lambda(self, max_iters=100, n_folds=3, max_evals=100, max_lambda=10, optimizer=DEFAULT_OPTIMIZER):
        self.cv_indices = KFold(self.x.shape[0], n_folds=n_folds, shuffle=True)
        self.cross_max_iters = max_iters
        space = hp.choice('model', [{
            'lambda_1': hp.uniform('lambda_1', 0, max_lambda),
            'lambda_2': hp.uniform('lambda_2', 0, max_lambda)
        }])

        best = fmin(self.__cross_validation,
                    space=space, algo=tpe.suggest,
                    max_evals=max_evals
                    )
        self.lambda_1 = best['lambda_1']
        self.lambda_2 = best['lambda_2']
        return best

    def set_lambda_no_search(self, max_iters=100, n_folds=3, lambda_1_list=None, lambda_2_list=None):
        self.cv_indices = KFold(self.x.shape[0], n_folds=n_folds, shuffle=True)
        self.cross_max_iters = max_iters
        pars = []
        mean_scores = []
        std_scores = []
        for l_1 in lambda_1_list:
            for l_2 in lambda_2_list:
                params = {'lambda_1':l_1, 'lambda_2':l_2}
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
        self.lambda_1 = best['lambda_1']
        self.lambda_2 = best['lambda_2']
        self.path = {'params':pars, 'scores':mean_scores, 'std_scores': std_scores}
        return best

    def get_alpha(self):
        return {'lambda_1': self.lambda_1, 'lambda_2': self.lambda_2}


