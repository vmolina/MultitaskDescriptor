import time

import numpy as np
import scipy as sp
import tables
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel


def assign_task(n, t, task):
    for i in range(t):
        task[i * n:(i + 1) * n, i] = 1.0


def regress(x, beta, task):
    beta_ = task.dot(beta)
    y = np.sum(beta_ * x, axis=1)
    y.shape = (y.shape[0], 1)
    return y


def generate_dataset(tasks_n, n_tr, n_te, p, desc_dim, clusters=1, noise=.1, normalized=True, n_thetas=1, n_alphas=1,
                     beta_randomness=0):
    """
        tasks_n:   Number of tasks
        n_tr:  Number of training samples
        n_te:  Number of testing samples
        p:  Number of Dimensions
        desc_dim:   Number of Descriptors
        clusters: number of correlated features
        noise: noise to add to the prediction
        normalized: weather to normalize y or not
        n_thetas : dimension of theta_params
        n_alphas: number of alphas
        beta_randomness: number of randomly activated betas
    """
    descriptors_loc = []
    descriptors_scale = []

    selected_thetas = np.random.choice(range(p), size=n_thetas, replace=False)
    theta = np.zeros((1, p), dtype=np.float64)
    alpha = np.zeros((p, desc_dim), dtype=np.float64)

    for i in selected_thetas:
        theta[0, i] = np.random.gamma(scale=1, shape=2)
        selected_alphas = np.random.choice(range(desc_dim), size=n_alphas, replace=False)
        for j in selected_alphas:
            alpha[i, j] = np.random.normal(0, 2)

    for cluster in range(clusters):
        descriptors_loc.append(np.random.normal(loc=0, scale=5, size=desc_dim))
        descriptors_scale.append(np.random.gamma(scale=.2, shape=1., size=desc_dim))
    descriptors = np.zeros((tasks_n, desc_dim), dtype=np.float64)
    for t in xrange(tasks_n):
        for i in xrange(desc_dim):
            j = t % clusters
            descriptors[t, i] = np.random.normal(loc=descriptors_loc[j][i], scale=descriptors_scale[j][i])

    sigma = np.linalg.inv(sp.stats.wishart.rvs(p+20,np.eye(p)))


    x_tr = np.zeros((n_tr * tasks_n, p), dtype=np.float64)
    x_tr[:] = np.random.multivariate_normal(np.zeros((p)),sigma, size=(n_tr * tasks_n))
    x_te = np.zeros((n_te * tasks_n, p), dtype=np.float64)
    x_te[:] = np.random.multivariate_normal(np.zeros((p)),sigma, size=(n_te * tasks_n))

    # Task matrices
    task_tr = np.zeros((n_tr * tasks_n, tasks_n), dtype=np.float64)
    assign_task(n_tr, tasks_n, task_tr)
    task_te = np.zeros((n_te * tasks_n, tasks_n), dtype=np.float64)
    assign_task(n_te, tasks_n, task_te)
    beta = theta * alpha.dot(descriptors.T).T

    beta /= np.max(np.abs(beta).flatten())


    for i in range(beta_randomness):
        i = np.random.choice(range(beta.shape[0]))
        j = np.random.choice(range(beta.shape[1]))

        if beta[i, j] == 0:
            beta[i, j] = np.random.normal(scale=.5)
        else:
            beta[i, j] = 0.

    if noise:
        y_tr = regress(x_tr, beta, task_tr) + np.random.normal(scale=.1, size=(x_tr.shape[0], 1))
    else:
        y_tr = regress(x_tr, beta, task_tr)

    y_te = regress(x_te, beta, task_te)

    if normalized:
        max_ = np.max(np.abs(y_tr.flatten()))
        y_tr /= max_
        y_te /= max_


    return descriptors, alpha, theta, x_tr, x_te, task_tr, task_te, y_tr, y_te, beta


def run_simulation(filename, experiments, derived_descriptors=False, random_descriptors=False):
    """
    filename is a string corresponding to the filename where the results can be kept
    Experiment is a list of dictionaries with the following fields:
    {
        file        :   file to save the results
        params : {
            T           :   Number of tasks
            n_tr        :   Number of training samples
            n_te        :   Number of testing samples
            p           :   Number of Dimensions
            D           :   Number of Descriptors
        }
        name        :   Name of experiment
        repetitions :   Number of repetitions
        methods     :   A list of  tuples containing method's name, method's class,  a dictionary of arguments
                        and True or False to indicate whether descriptors should be passed.
    }
    """
    if derived_descriptors and random_descriptors:
        raise Exception("derived_descriptors and random_descriptors cannot be True at the same time")
    h5 = tables.open_file(filename, 'a')
    root = h5.root

    for i, experiment in enumerate(experiments):

        print 'Experiment: %i of %i' % (i, len(experiments))
        experiment_group = h5.create_group(root, experiment['name'])
        for param in experiment['params']:
            h5.set_node_attr(experiment_group, param, experiment['params'][param])
        h5.set_node_attr(experiment_group, 'repetitions', experiment['repetitions'])
        h5.set_node_attr(experiment_group, 'methods', experiment['methods'])
        experiment_path = experiment_group._v_pathname
        # h5.close()
        #Parallel(n_jobs=3)(delayed(repit)(filename,experiment_path, experiment, repetition)
        #    for repetition in xrange(experiment['repetitions']))
        for repetition in xrange(experiment['repetitions']):
            repit(h5, experiment_path, experiment, repetition, derived_descriptors, random_descriptors)
    h5.close()


def repit(h5, experiment_group, experiment, repetition, derived_descriptors, random_descriptors,
          verbose=True):
    # h5 = tables.open_file(h5_filename, 'a')
    rep_group = h5.create_group(experiment_group, "repetition_%i" % repetition)
    params = experiment['params']
    descriptors, alpha, theta, x_tr, x_te, task_tr, task_te, y_tr, y_te, beta= generate_dataset(**params)
    h5.create_carray(rep_group, 'descriptors', obj=descriptors)
    h5.create_carray(rep_group, 'alpha', obj=alpha)
    h5.create_carray(rep_group, 'theta', obj=theta)
    h5.create_carray(rep_group, 'x_tr', obj=x_tr)
    h5.create_carray(rep_group, 'x_te', obj=x_te)
    h5.create_carray(rep_group, 'task_tr', obj=task_tr)
    h5.create_carray(rep_group, 'task_te', obj=task_te)
    h5.create_carray(rep_group, 'y_tr', obj=y_tr)
    h5.create_carray(rep_group, 'y_te', obj=y_te)
    h5.create_carray(rep_group, 'beta', obj=beta)

    if derived_descriptors:
        derived_descriptors = rbf_kernel(descriptors)
        h5.create_carray(rep_group, 'derived_descriptors', obj=derived_descriptors)
    elif random_descriptors:
        derived_descriptors = np.random.uniform(size=descriptors.shape)
        h5.create_carray(rep_group, 'random_descriptors', obj=derived_descriptors)
    else:
        derived_descriptors = descriptors

    for meth_name, method, meth_params, desc_bool, alpha_params, max_iters in experiment['methods']:
        print '\t', repetition, meth_name
        meth_group = h5.create_group(rep_group, meth_name)

        if desc_bool:
            model = method(x_tr, y_tr, task_tr, derived_descriptors, **meth_params)
            ts = time.time()
            model.set_alpha(**alpha_params)
            model.optimize(max_iters=max_iters)
            te = time.time()

        else:
            model = method(x_tr, y_tr, task_tr, **meth_params)
            ts = time.time()
            model.set_alpha(**alpha_params)
            model.optimize(max_iters=max_iters)
            te = time.time()

        training_time = te - ts
        #h5.set_node_attr(meth_group, 'iterations', model.iterations)
        h5.set_node_attr(meth_group, 'training_time', training_time)
        alphas = model.get_alpha()
        for alpha in alphas:
            h5.set_node_attr(meth_group, alpha, alphas[alpha])

        params = model.get_params()
        for param in params:
            h5.create_carray(meth_group, param, obj=params[param])
        if desc_bool:
            y_pr = model.predict(x_te, task_te, derived_descriptors)
        else:
            y_pr = model.predict(x_te, task_te)

        h5.create_carray(meth_group, 'y_pred', obj=y_pr)
        #h5.close()


def grab_results(filename, method_names, random_descriptors=False):
    h5 = tables.open_file(filename)
    results = {}
    for sim_name in h5.root._v_children:
        sim = h5.root._v_children[sim_name]
        sim_results = {}
        sim_results['n_tr'] = h5.get_node_attr(sim, 'n_tr')
        sim_results['n_te'] = h5.get_node_attr(sim, 'n_te')
        sim_results['p'] = h5.get_node_attr(sim, 'p')
        sim_results['T'] = h5.get_node_attr(sim, 'tasks_n')
        sim_results['D'] = h5.get_node_attr(sim, 'desc_dim')


        for method_name in method_names:
            method_results = {
                'time': [], 'mse': [], 'intersection': [], 'union': [],
                'truth_beta': [], 'truth': [], 'beta': []
            }

            for rep_name in np.sort(sim._v_children.keys()):

                rep = sim._v_children[rep_name]
                descriptors = h5.get_node(rep, 'descriptors')

                try:
                    truth_beta = h5.get_node(rep, 'beta')

                except tables.NoSuchNodeError:
                    theta = h5.get_node(rep, 'theta')
                    alpha = h5.get_node(rep, 'alpha')
                    truth_beta = (theta * np.dot(descriptors, np.array(alpha).T))

                beta_mask = truth_beta != 0

                method = h5.get_node(rep, method_name)
                method_results['time'].append(h5.get_node_attr(method, 'training_time'))
                # method_results['iterations'].append(h5.get_node_attr(method, 'iterations'))

                y_pred = h5.get_node(method, 'y_pred')
                y_te = h5.get_node(rep, 'y_te')
                if random_descriptors:
                    descriptors = h5.get_node(rep, 'random_descriptors')
                try:
                    beta = np.array(h5.get_node(method, 'beta'))
                except tables.NoSuchNodeError:
                    theta = np.array(h5.get_node(method, 'theta'))
                    if 'theta' in method_results:
                        method_results['theta'].append(theta)
                    else:
                        method_results['theta'] = [theta]
                    try:
                        gamma = np.array(h5.get_node(method, 'gamma'))
                        if 'gamma' in method_results:
                            method_results['gamma'].append(gamma)
                        else:
                            method_results['gamma'] = [gamma]
                        beta = theta*gamma
                    except tables.NoSuchNodeError:
                        alpha = np.array(h5.get_node(method, 'alpha'))
                        if 'alpha' in method_results:
                            method_results['alpha'].append(alpha)
                        else:
                            method_results['alpha'] = [alpha]
                        beta = theta * np.dot(descriptors, alpha.T)


                beta_method_mask = beta != 0
                method_results['intersection'].append(np.sum(beta_method_mask*beta_mask)/float(np.sum(beta_mask)))
                method_results['union'].append(np.sum((beta_method_mask+beta_mask)>0)/float(np.sum(beta_mask)))
                method_results['truth_beta'].append(truth_beta)
                method_results['truth'].append(beta_mask)
                method_results['beta'].append(beta)

                method_results['mse'].append(np.sqrt(mean_squared_error(y_te, y_pred)) / (np.max(y_te) - np.min(y_te)))

            sim_results[method_name] = method_results
        results[sim_name] = sim_results

    h5.close()
    return results
