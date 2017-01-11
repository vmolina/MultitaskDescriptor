from __future__ import division

from sklearn.cross_validation import KFold
from sklearn.linear_model import Lasso, LassoLars
from hyperopt import fmin, tpe, hp

from mulas import MSE
from lasso_wrappers import *


class MuMuLaDer(object):
    theta = None
    alpha = None

    def __init__(self, x, y, tasks, descriptors, intercept=False, task_intercept=False, lambda_1=None, lambda_2=None,
                 alpha_params=None, theta_params=None, tol=0.01, random_init=False, loss='square', regul= 'l1', loss_f=MSE):
        """
        Define the model. It does not uses the traditional format by SciKit but the one from GPy
        x: A numpy array of shape (n,p) for the independent variables.
        y: A numpy array of haspe (n,1) for the dependent variables.
        tasks: A numpy array with (n,T) where every row contain just
            one element 1 and 0 for the other numbers to indicate to
            which task belongs.
        descriptors: An (T,L) array containing the different descriptors
            or each task.
        intercept: If True an intercept variable is used
        task_intercept: If True an intercept variable is added tot ask descriptors
        lambda_1: Regularization parameter for theta
        lambda_2: Regularization parameter for alpha
        alpha_params: dictionary containing the parameters for the alpha optimizer
        theta_params: dictionary containing the parameter fo the theta optimizer
        tol: float. if the change in beta is smaller than this number it will stop the optimization.
        random_init: weather to use random initialization or not
        loss: which loss function to use (only for SPAMS)
        regul: which regularization to use (only for SPAMS)
        loss_f: loss function to minimize in cross_validation
        epsilon: A parameter for defining when to stop the optimization.
        """
        self.intercept = intercept
        self.task_intercept = task_intercept

        self.x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=x.dtype)), axis=1) if self.intercept else x
        self.y = y
        self.tasks = tasks
        self.descriptors = np.concatenate((descriptors,
                                           np.ones((descriptors.shape[0], 1.), dtype=descriptors.dtype)),
                                          axis=1) if self.task_intercept else descriptors
        self.lambda_1 = 1. if lambda_1 is None else lambda_1
        self.lambda_2 = 1. if lambda_2 is None else lambda_2
        self.theta = None
        self.alpha = None
        self.alpha_params = {} if alpha_params is None else alpha_params
        self.theta_params = {} if theta_params is None else theta_params
        self.tol = tol
        self.iterations = 0
        self.random_init = random_init

        self.loss = loss
        self.regul = regul
        self.loss_f = loss_f

    def optimize(self, optimizer=DEFAULT_OPTIMIZER, max_iters=100, max_inner_iters=10):
        """
        It fits the model.
        max_iters: maximum iterations
        max_inner_itters: how many iterations to use in every lasso optimization
        optimizer: Select the algorithm for optimize.
            'Lasso' uses the Lasso implementation of sklearn
            'LassoLars' uses the LassoLars implementation of sklearn
            'spams' uses the Lasso implementation of spams

        """
        # get input sizes and initizialize parameters
        n, p = self.x.shape
        T = self.tasks.shape[1]
        L = self.descriptors.shape[1]

        # Precalculate x_alpha matrix for optimizing alpha
        # It is a (n, p*L) where xD_{i,(j-1)L+l) = \theta_{j}d_l for the
        # corresponding task.

        # n_T = list(np.sum(tasks, axis=0))
        repeated_descriptors = np.repeat(self.descriptors, p, axis=0)
        repeated_descriptors.shape = (T, L * p)
        x_alpha = self.tasks.dot(repeated_descriptors) * np.repeat(self.x, L, axis=1)
        del repeated_descriptors
        # Initialize theta
        self.theta = np.ones((1, p), dtype=np.float64)
        if optimizer == LASSO:
            optimize_theta = Lasso(warm_start=True, positive=True, alpha=self.lambda_1, **self.theta_params)
            optimize_alpha = Lasso(warm_start=True, alpha=self.lambda_2, **self.alpha_params)
        elif optimizer == LASSOLARS:
            optimize_theta = LassoLars(alpha=self.lambda_1, **self.theta_params)
            optimize_alpha = LassoLars(alpha=self.lambda_2, **self.alpha_params)
        elif optimizer == LASSOSPAMS:
            optimize_theta = SpamWrapperOptimizer(positive=True, lambda_=self.lambda_1,
                                                  params=self.theta_params, loss=self.loss, regul=self.regul)
            optimize_alpha = SpamWrapperOptimizer(lambda_=self.lambda_2, params=self.alpha_params,
                                                  loss=self.loss, regul=self.regul )
        else:
            raise Exception("Not a valid value")

        if self.random_init:
            self.alpha = np.random.normal(size=(p, L)).astype(np.float64)
            optimize_alpha.coef_ = self.alpha.flatten()
        else:
            self.alpha = np.zeros((p, L), dtype=np.float64)
            optimize_alpha.coef_ = self.alpha.flatten()

        beta_0 = self.get_beta(self.descriptors)

        x_theta = self.tasks.dot(self.descriptors.dot(self.alpha.T)) * self.x
        # Start the two phase optimization
        continue_optimization = True
        self.iterations = 0
        while continue_optimization:
            # Optimize for alpha

            self.alpha[:] = optimize_alpha.fit(
                np.repeat(self.theta, L, axis=1) * x_alpha,
                self.y, iters=max_inner_iters).coef_.reshape((p, L))

            beta = self.get_beta(self.descriptors)

            # Optimize for theta
            x_theta[:] = self.tasks.dot(self.descriptors.dot(self.alpha.T)) * self.x
            self.theta[:] = optimize_theta.fit(x_theta, self.y, iters=max_inner_iters).coef_
            # self.theta.shape = (1, len(self.theta))

            self.iterations += 1
            # print 'Mumulader', self.iterations
            # f = plt.figure()
            # plt.matshow(beta - beta_0)
            # plt.colorbar()
            # plt.show()
            # print "Norm: {}".format(np.linalg.norm(beta.flatten() - beta_0.flatten()))
            if np.linalg.norm(beta.flatten() - beta_0.flatten()) < self.tol:
                continue_optimization = False
            else:
                beta_0 = beta
            if self.iterations >= max_iters:
                continue_optimization = False

    def get_beta(self, descriptors=None):
        if descriptors is None:
            descriptors=self.descriptors
        if self.descriptors.shape[1] == descriptors.shape[1]:
            desc_ = descriptors
        else:
            desc_ = np.concatenate((descriptors,
                                    np.ones((descriptors.shape[0], 1), dtype=descriptors.dtype)),
                                   axis=1) if self.task_intercept else descriptors

        return self.theta * desc_.dot(self.alpha.T)

    def predict(self, x, tasks, descriptors):

        new_x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=x.dtype)), axis=1) if self.intercept else x

        new_descriptors = np.concatenate((descriptors,
                                          np.ones((descriptors.shape[0], 1), dtype=descriptors.dtype)),
                                         axis=1) if self.task_intercept else descriptors
        beta = self.get_beta(new_descriptors)
        return np.sum(tasks.dot(beta) * new_x, axis=1)

    def get_params(self):
        return {
            'theta': self.theta,
            'alpha': self.alpha
        }

    def __cross_validation(self, values, std=False):
        lambda_1 = values['lambda_1']
        lambda_2 = values['lambda_2']

        mse = []
        for train, test in self.cv_indices:
            x_tr = self.x[train, :]
            y_tr = self.y[train]
            x_te = self.x[test, :]
            y_te = self.y[test]
            tasks_tr = self.tasks[train, :]
            tasks_te = self.tasks[test, :]
            descriptors = self.descriptors
            model = MuMuLaDer(x_tr, y_tr, tasks_tr, descriptors, lambda_1=lambda_1, lambda_2=lambda_2,
                              intercept=False, task_intercept=False,
                              alpha_params=self.alpha_params,
                              theta_params=self.theta_params, tol=self.tol, random_init=self.random_init)
            model.optimize(max_iters=self.cross_max_iters)
            y_pr = model.predict(x_te, tasks_te, self.descriptors)
            mse.append(self.loss_f(y_te.flatten(), y_pr.flatten()))
        if std:
            return np.mean(mse), np.std(mse)
        return np.mean(mse)

    def set_lambda(self, max_iters=100, n_folds=3, max_evals=100, max_lambda=10):
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
                if  par[key] < max_par[key]:
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


def regress(x, descriptors, alpha, theta, task):
    y = np.zeros((x.shape[0], 1), dtype=np.float64)
    beta = task.dot(theta * alpha.dot(descriptors.T).T)
    y[:,0] = np.sum(beta * x, axis=1)

    return y


def assign_task(n, T, task):
    for i in range(T):
        task[i * n:(i + 1) * n, i] = 1.0
