from __future__ import division

from sklearn.cross_validation import KFold
from sklearn.linear_model import Lasso, LassoLars
from sklearn.metrics import mean_squared_error as MSE
from hyperopt import fmin, tpe, hp

from lasso_wrappers import *


class MultitaskLassoDescriptors(object):
    theta = None
    alpha = None

    def __init__(self, x, y, tasks, descriptors, intercept=False, task_intercept=False, lambda_1=None, lambda_2=None,
                 alpha_params=None, theta_params=None, tol=0.001, random_init=False):

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

    def optimize(self, optimizer=DEFAULT_OPTIMIZER, max_iters=100):
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
            optimize_theta = SpamWrapperOptimizer(positive=True, lambda_=self.lambda_1, params=self.theta_params)
            optimize_alpha = SpamWrapperOptimizer(lambda_=self.lambda_2, params=self.alpha_params)
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
                self.y).coef_.reshape((p, L))

            beta = self.get_beta(self.descriptors)

            # Optimize for theta
            x_theta[:] = self.tasks.dot(self.descriptors.dot(self.alpha.T)) * self.x
            self.theta[:] = optimize_theta.fit(x_theta, self.y).coef_


            self.iterations += 1

            if np.linalg.norm(beta.flatten() - beta_0.flatten()) < self.tol:
                continue_optimization = False
            else:
                beta_0 = beta
            if self.iterations >= max_iters:
                continue_optimization = False

    def get_beta(self, descriptors):
        """It calculates the beta matrix for the given matrix of descriptors.
           descriptors: A matrix of descriptors with the same number of descriptors as the original matrix.
        """
        if self.descriptors.shape[1] == descriptors.shape[1]:
            desc_ = descriptors
        elif self.descriptors.shape[1] == descriptors.shape[1]+1:
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

    def __cross_validation(self, values):
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
            mse.append(MSE(y_te, y_pr))

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
        scores = []
        for l_1 in lambda_1_list:
            for l_2 in lambda_2_list:
                params = {'lambda_1':l_1, 'lambda_2': l_2}
                score = self.__cross_validation(params)
                scores.append(score)
                pars.append(params)
        best = pars[np.argmin(scores)]
        self.lambda_1 = best['lambda_1']
        self.lambda_2 = best['lambda_2']
        self.path = {'params':pars, 'scores':scores}
        return best

    def get_lambda(self):
        return {'lambda_1': self.lambda_1, 'lambda_2': self.lambda_2}


def regress(x, descriptors, alpha, theta, task):
    y = np.zeros((x.shape[0], 1), dtype=np.float64)
    beta = task.dot(theta * alpha.dot(descriptors.T).T)
    y[:] = np.sum(beta * x, axis=1)

    return y


def assign_task(n, T, task):
    for i in range(T):
        task[i * n:(i + 1) * n, i] = 1.0


if __name__ == '__main__':
    T = 4  # Number of tasks
    n_tr = 100  # Number of training samples
    n_te = 10  # Number of testing samples
    p = 10  # Number of Dimensions
    D = 3  # Number of Descriptors

    descriptors = np.random.standard_normal(size=(T, D))
    alpha = np.random.standard_normal(size=(p, D))
    theta = np.random.standard_gamma(shape=1, size=(1, p))
    theta[0, 3:] = 0
    x_tr = np.random.standard_normal(size=(n_tr * (T - 1), p))
    x_te = np.random.standard_normal(size=(n_te * T, p))

    # Task matrices

    task_tr = np.zeros((n_tr * (T - 1), T))
    assign_task(n_tr, T - 1, task_tr)
    task_te = np.zeros((n_te * T, T))
    assign_task(n_te, T, task_te)

    y_tr = regress(x_tr, descriptors, alpha, theta, task_tr)
    y_te = regress(x_te, descriptors, alpha, theta, task_te)
    model = MultitaskLassoDescriptors(x_tr, y_tr, task_tr, descriptors)
    model.optimize()
    print model.get_beta(descriptors)
