"""
This file contains functions for performing running fair regression
algorithms and the set of baseline methods.

See end of file to see sample use of running fair regression.
"""

from __future__ import print_function

import functools
import numpy as np
import pandas as pd
import data_parser as parser
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import pickle
import eval as evaluate
import solvers as solvers
import exp_grad as fairlearn
from collections import Counter
import math
print = functools.partial(print, flush=True)




# Global Variables
TEST_SIZE = 0.5  # fraction of observations from each protected group
Theta = np.linspace(0, 1.0, 41)
eta = (Theta[1] - Theta[0])/2
DATA_SPLIT_SEED = 4
_SMALL = True  # small scale dataset for speed and testing
_CORESET = True


def coresetSample(x, a, y):
    """
    Randomly subsample a smaller dataset of certain size
    """
    # gives sizes for each group
    print("starting to get the coreset....")
    AkSizes = Counter(a)
    # AkSizes[0] and AkSizes[1]

    lGroups = len(AkSizes)

    print("computing svd now...")
    # svd to get leverage scores
    u,s,v = np.linalg.svd(x, full_matrices=False)
    sqrtLevScores = np.linalg.norm(u, 2, 1)
    levScores = np.square(sqrtLevScores)
    print("Got the svd...")


    d = sum(levScores)
    logl = math.log(lGroups, 2)
    log1ByEta = math.log(1.0/eta, 2)

    tpart = d + logl + log1ByEta
    invAkSizes = np.array([1.0/AkSizes[x] for x in a])
    tprob = tpart * invAkSizes

    samplingProb = levScores + tprob

    # rvals = [2**i for i in range(0, int(math.log(x.shape[0], 2)-1))]
    rvals = [i for i in range(1,2)]

    weights = {}
    sampledX = {}
    sampledA = {}
    sampledY = {}

    weightsOut = []


    print("Going for sampling...")
    for j in range(len(rvals)):

        r = rvals[j]
        weights[r] = []
        sampledX[r] = pd.DataFrame(columns=x.columns)
        sampledA[r] = pd.Series([])
        sampledY[r] = pd.Series([])

        uniqPointsSampled = 0

        for i in range(len(x)):
            tossProb = min(r*samplingProb[i], 1)
            uniSample =  np.random.random_sample()

            if uniSample <= tossProb:
                uniqPointsSampled += 1
                weight_i = 1.0/tossProb
                weights[r].append(weight_i)
                
                weight_i = math.floor(weight_i)

                if weight_i < 1:
                    weight_i = 1

                # weightsOut.append(weight_i)

                # add point weight many times
                for w in range(weight_i):
                    sampledX[r] = sampledX[r].append(x.iloc[i])
                    sampledA[r] = sampledA[r].append(pd.Series([a[i]]))
                    sampledY[r] = sampledY[r].append(pd.Series([y[i]]))

        sampledX[r].index = range(len(sampledX[r]))
        sampledA[r].index = range(len(sampledX[r]))
        sampledY[r].index = range(len(sampledX[r]))


        print("num Points Sampled -- for r = ", r, " and uniq num points = ", uniqPointsSampled)
        print("num Points Sampled -- for r = ", r, " and num points = ", len(sampledX[r]))


    # for i in weightsOut:
        # print(i)

    return sampledX, sampledA, sampledY, uniqPointsSampled


def subsample(x, a, y, size, random_seed=DATA_SPLIT_SEED):
    """
    Randomly subsample a smaller dataset of certain size
    """
    toss = 1 - size / (len(x))
    x1, _, a1, _, y1 ,_ = train_test_split(x, a, y, test_size=toss, random_state=random_seed)
    x1.index = range(len(x1))
    y1.index = range(len(x1))
    a1.index = range(len(x1))
    return x1, a1, y1


def fair_train_test(dataset, size, eps_list, learner, constraint="DP",
                   loss="square", random_seed=DATA_SPLIT_SEED, init_cache=[]):
    """
    Input:
    - dataset name
    - size parameter for data parser
    - eps_list: list of epsilons for exp_grad
    - learner: the solver for CSC
    - constraint: fairness constraint name
    - loss: loss function name
    - random_seed

    Output: Results for
    - exp_grad: (eps, loss) for training and test sets
    - benchmark method: (eps, loss) for training and test sets
    """

    if dataset == 'law_school':
        x, a, y = parser.clean_lawschool_full()
    elif dataset == 'communities':
        x, a, y = parser.clean_communities_full()
    elif dataset == 'adult':
        x, a, y = parser.clean_adult_full()
    else:
        raise Exception('DATA SET NOT FOUND!')
  
    if _SMALL:
        x, a, y = subsample(x, a, y, size)
    

    # x_train, a_train, y_train, x_test, a_test, y_test = train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)
    x_train, a_train, y_train = x, a, y
    x_train.index = range(len(x))
    y_train.index = range(len(y))
    a_train.index = range(len(a))

    if _CORESET:
        xcor, acor, ycor, uniqSamples = coresetSample(x_train, a_train, y_train)
        print("got the coreset")

    if _UNIFORM:
        xuni, auni, yuni, uniqSamples = uniformSample(x_train, a_train, y_train)
        print("got the coreset")        

    fair_model = {}
    train_evaluation = {}
    cor_fair_model = {}
    cor_evaluation = {}
    
    train_cor_evaluation = {}
    coreset_train_evaluation = {}
    
    # test_evaluation = {}

    # for r
    for eps in eps_list:
        fair_model[eps] = fairlearn.train_FairRegression(x_train, a_train, y_train, eps, Theta, learner, constraint, loss, init_cache=init_cache)

        train_evaluation[eps] = evaluate.evaluate_FairModel(x_train, a_train, y_train, loss, fair_model[eps]['exp_grad_result'], Theta)

        # test_evaluation[eps] = evaluate.evaluate_FairModel(x_test, a_test, y_test, loss, fair_model[eps]['exp_grad_result'], Theta)

        print("Running on the coreset....")

        cor_fair_model[eps] = fairlearn.train_FairRegression(xcor[1], acor[1], ycor[1], eps, Theta, learner, constraint, loss, init_cache=init_cache)

        cor_evaluation[eps] = evaluate.evaluate_FairModel(xcor[1], acor[1], ycor[1], loss, cor_fair_model[eps]['exp_grad_result'], Theta)

        train_cor_evaluation[eps] = evaluate.evaluate_FairModel(xcor[1], acor[1], ycor[1], loss, fair_model[eps]['exp_grad_result'], Theta)

        coreset_train_evaluation[eps] = evaluate.evaluate_FairModel(x_train, a_train, y_train, loss, cor_fair_model[eps]['exp_grad_result'], Theta)

        print("done with epsilon", eps)

    result = {}
    result['dataset'] = dataset
    result['learner'] = learner.name
    result['loss'] = loss
    result['constraint'] = constraint
    result['train_eval'] = train_evaluation
    result['cor_eval'] = cor_evaluation
    result['coreset_train_eval'] = coreset_train_evaluation
    result['train_coreset_eval'] = train_cor_evaluation
    # result['test_eval'] = test_evaluation
    return result


def base_train_test(dataset, size, base_solver, loss="square",
                    random_seed=DATA_SPLIT_SEED):
    """
    Given a baseline method, train and test on a dataset.

    Input:
    - dataset name
    - size parameter for data parser
    - base_solver: the solver for baseline benchmark
    - loss: loss function name
    - random_seed for data splitting

    Output: Results for
    - baseline output
    """
    if dataset == 'law_school':
        x, a, y = parser.clean_lawschool_full()
        sens_attr = 'race'
    elif dataset == 'communities':
        x, a, y = parser.clean_communities_full()
        sens_attr = 'race'
    elif dataset == 'adult':
        x, a, y = parser.clean_adult_full()
        sens_attr = 'sex'
    else:
        raise Exception('DATA SET NOT FOUND!')


    if _SMALL:
        x, a, y = subsample(x, a, y, size)

    x_train, a_train, y_train, x_test, a_test, y_test = train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)

    if base_solver.name == "SEO":
        # Evaluate SEO method
        base_solver.fit(x_train, y_train, sens_attr)
        h_base = lambda X: base_solver.predict(X, sens_attr)
    else:
        base_solver.fit(x_train, y_train)
        h_base = lambda X: base_solver.predict(X)

    base_train_eval = evaluate.eval_BenchmarkModel(x_train, a_train,
                                                   y_train, h_base,
                                                   loss)
    base_test_eval = evaluate.eval_BenchmarkModel(x_test, a_test,
                                                  y_test, h_base,
                                                  loss)
    result = {}
    result['base_train_eval'] = base_train_eval
    result['base_test_eval'] = base_test_eval
    result['loss'] = loss
    result['learner'] = base_solver.name
    result['dataset'] = dataset
    return result


def square_loss_benchmark(dataset, n):
    """
    Run the set of unconstrained methods for square loss
    OLS_Base_Learner
    RF_Base_Regressor
    XGB_Base_Regressor
    """
    loss = 'square'
    base_solver1 = solvers.OLS_Base_Learner()
    base_res1 = base_train_test(dataset, n, base_solver1, loss=loss,
                                random_seed=DATA_SPLIT_SEED)

    base_solver4 = solvers.SEO_Learner()
    base_res4 = base_train_test(dataset, n, base_solver4, loss=loss,
                                random_seed=DATA_SPLIT_SEED)

    if _SMALL:
        bl = [base_res1, base_res4]
    else:
        base_solver2 = solvers.RF_Base_Regressor(max_depth=4,
                                                 n_estimators=200)
        base_res2 = base_train_test(dataset, n, base_solver2,
                                    loss=loss,
                                    random_seed=DATA_SPLIT_SEED)

        base_solver3 = solvers.XGB_Base_Regressor(max_depth=4,
                                                  n_estimators=200)
        base_res3 = base_train_test(dataset, n, base_solver3,
                                    loss=loss,
                                    random_seed=DATA_SPLIT_SEED)
        
        bl = [base_res1, base_res2, base_res3, base_res4]
    return bl


def log_loss_benchmark(dataset='adult', size=100):
    """
    Run the set of unconstrained methods for logistic loss
    LogisticRegression
    XGB_Base_Classifier
    """
    loss = 'logistic'
    base_solver1 = solvers.Logistic_Base_Learner(C=10)
    base_res1 = base_train_test(dataset, size, base_solver1, loss=loss,
                                random_seed=DATA_SPLIT_SEED)
    print("Done with Logistic base")

    if _SMALL:
        bl = [base_res1]
    else:
        base_solver3 = solvers.XGB_Base_Classifier(max_depth=3,
                                                   n_estimators=150,
                                                   gamma=2)
        base_res3 = base_train_test(dataset, size, base_solver3, loss=loss,
                                random_seed=DATA_SPLIT_SEED)
        print("Done with XGB base")
        bl = [base_res1, base_res3]
    return bl




def read_result_list(result_list):
    """
    Parse the experiment a list of experiment result and print out info
    """

    '''
    result['dataset'] = dataset
    result['learner'] = learner.name
    result['loss'] = loss
    result['constraint'] = constraint
    result['train_eval'] = train_evaluation
    result['cor_eval'] = cor_evaluation
    result['coreset_train_eval'] = coreset_train_evaluation
    result['train_coreset_eval'] = train_cor_evaluation
    '''


    for result in result_list:
        learner = result['learner']
        dataset = result['dataset']
        train_eval = result['train_eval']
        cor_eval = result['cor_eval']
        coreset_train_eval = result['coreset_train_eval']
        train_coreset_eval = result['train_coreset_eval']
        # test_eval = result['test_eval']
        loss = result['loss']
        constraint = result['constraint']
        learner = result['learner']
        dataset = result['dataset']
        eps_vals = train_eval.keys()
        train_disp_dic = {}
        cor_disp_dic = {}
        cor_train_disp_dic = {}
        train_cor_disp_dic = {}

        train_err_dic = {}
        cor_err_dic = {}
        cor_train_err_dic = {}
        train_cor_err_dic = {}
        
        test_disp_dic = {}
        test_err_dic = {}
        test_loss_std_dic = {}
        test_disp_dev_dic = {}

        for eps in eps_vals:
            train_disp = train_eval[eps]["DP_disp"]
            cor_disp = cor_eval[eps]["DP_disp"]
            cor_train_disp = coreset_train_eval[eps]["DP_disp"]
            train_cor_disp = train_coreset_eval[eps]["DP_disp"]
            # test_disp = test_eval[eps]["DP_disp"]
            train_disp_dic[eps] = train_disp
            cor_disp_dic[eps] = cor_disp
            cor_train_disp_dic[eps] = cor_train_disp
            train_cor_disp_dic[eps] = train_cor_disp
            # test_disp_dic[eps] = test_disp

            # test_loss_std_dic[eps] = test_eval[eps]['loss_std']
            # test_disp_dev_dic[eps] = test_eval[eps]['disp_std']

            if loss == "square":
                # taking the RMSE
                train_err_dic[eps] = np.sqrt(train_eval[eps]['weighted_loss'])
                cor_err_dic[eps] = np.sqrt(cor_eval[eps]['weighted_loss'])
                cor_train_err_dic[eps] = np.sqrt(coreset_train_eval[eps]['weighted_loss'])
                train_cor_err_dic[eps] = np.sqrt(train_coreset_eval[eps]['weighted_loss'])
                # test_err_dic[eps] = np.sqrt(test_eval[eps]['weighted_loss'])

            else:
                train_err_dic[eps] = (train_eval[eps]['weighted_loss'])
                cor_err_dic[eps] = (cor_train_eval[eps]['weighted_loss'])
                cor_train_err_dic[eps] = (coreset_train_eval[eps]['weighted_loss'])
                train_cor_err_dic[eps] = (train_coreset_eval[eps]['weighted_loss'])
                # test_err_dic[eps] = (test_eval[eps]['weighted_loss'])

        # taking the pareto frontier
        train_disp_list = [train_disp_dic[k] for k in eps_vals]
        cor_disp_list = [cor_disp_dic[k] for k in eps_vals]
        cor_train_disp_list = [cor_train_disp_dic[k] for k in eps_vals]
        train_cor_disp_list = [train_cor_disp_dic[k] for k in eps_vals]
        # test_disp_list = [test_disp_dic[k] for k in eps_vals]
        train_err_list = [train_err_dic[k] for k in eps_vals]
        cor_err_list = [cor_err_dic[k] for k in eps_vals]
        cor_train_err_list = [cor_train_err_dic[k] for k in eps_vals]
        train_cor_err_list = [train_cor_err_dic[k] for k in eps_vals]
        # test_err_list = [test_err_dic[k] for k in eps_vals]

        if loss == "square":
            show_loss = 'RMSE'
        else:
            show_loss = loss


        info = str('Dataset: '+dataset + '; loss: ' + loss + '; Solver: '+ learner)
        print(info)

        train_data = {'specified epsilon': list(eps_vals), 'SP disparity':train_disp_list, show_loss : train_err_list}
        train_performance = pd.DataFrame(data=train_data)

        cor_data = {'specified epsilon': list(eps_vals), 'SP disparity':cor_train_disp_list, show_loss : cor_err_list}
        cor_performance = pd.DataFrame(data=cor_data)


        cor_train_data = {'specified epsilon': list(eps_vals), 'SP disparity':cor_train_disp_list, show_loss : cor_train_err_list}
        cor_train_performance = pd.DataFrame(data=cor_train_data)  

        train_cor_data = {'specified epsilon': list(eps_vals), 'SP disparity':train_cor_disp_list, show_loss : train_cor_err_list}
        train_cor_performance = pd.DataFrame(data=train_cor_data)        
        # test_data = {'specified epsilon': list(eps_vals), 'SP disparity':test_disp_list, show_loss : test_err_list}
        # test_performance = pd.DataFrame(data=test_data)

        # Print out experiment info.
        print('Train set trade-off:')
        print(train_performance)
        print('Coreset set trade-off:')
        print(cor_performance)
        print('Coreset-Train set trade-off:')
        print(cor_train_performance)
        print('Train-Coreset set trade-off:')
        print(train_cor_performance)
        # print('Test set trade-off:')
        # print(test_performance)



# Sample instantiation of running the fair regeression algorithm
# eps_list = [0.275, 0.31, 1] # range of specified disparity values
eps_list = [0.1, 0.15] # range of specified disparity values

n = 200  # size of the sub-sampled dataset, when the flag SMALL is True
dataset = 'adult'  # name of the data set
constraint = "DP"  # name of the constraint; so far limited to demographic parity (or statistical parity)
loss = "square"  # name of the loss function
learner = solvers.LeastSquaresLearner(Theta) # Specify a supervised learning oracle oracle 

info = str('Dataset: '+dataset + '; loss: ' + loss + '; eps list: '+str(eps_list)) + '; Solver: '+learner.name
print('Starting experiment. ' + info)

# Run the fair learning algorithm the supervised learning oracle
result = fair_train_test(dataset, n, eps_list, learner,
                          constraint=constraint, loss=loss,
                          random_seed=DATA_SPLIT_SEED)

read_result_list([result])  # A simple print out for the experiment

# Saving the result list
outfile = open(info+'.pkl','wb')
pickle.dump(result, outfile)
outfile.close()

"""


# Other sample use:

learner1 = solvers.SVM_LP_Learner(off_set=alpha)
result1 = fair_train_test(dataset, n, eps_list, learner1,
                          constraint=constraint, loss=loss,
                          random_seed=DATA_SPLIT_SEED)

learner2 = solvers.LeastSquaresLearner(Theta)
result2 = fair_train_test(dataset, n, eps_list, learner2,
                          constraint=constraint, loss=loss,
                          random_seed=DATA_SPLIT_SEED)

learner3 = solvers.RF_Regression_Learner(Theta)
result3 = fair_train_test(dataset, n, eps_list, learner3,
                           constraint=constraint, loss=loss,
                           random_seed=DATA_SPLIT_SEED)

learner4 = solvers.XGB_Classifier_Learner(Theta)
result4 = fair_train_test(dataset, n, eps_list, learner4,
                           constraint=constraint, loss=loss,
                           random_seed=DATA_SPLIT_SEED)

learner5 = solvers.LogisticRegressionLearner(Theta)
result5 = fair_train_test(dataset, n, eps_list, learner5,
                          constraint=constraint, loss=loss,
                           random_seed=DATA_SPLIT_SEED)

learner6 = solvers.XGB_Regression_Learner(Theta)
result6 = fair_train_test(dataset, n, eps_list, learner6,
                          constraint=constraint, loss=loss,
                          random_seed=DATA_SPLIT_SEED)
"""

