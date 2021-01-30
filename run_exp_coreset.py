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
import sys
print = functools.partial(print, flush=True)


# Global Variables
TEST_SIZE = 0.5  # fraction of observations from each protected group
Theta = np.linspace(0, 1.0, 41)
eta = (Theta[1] - Theta[0])/2
DATA_SPLIT_SEED = 4
_SMALL = True       # small scale dataset for speed and testing
_CORESET = True     # fair coreset for the data
_UNIFORM = True     # uniform sampled dataset
_DUPLICATE = True   # to duplicate the sample points proportional to their weights
_TEST = True # whether you want to have test data or not


userR = float(sys.argv[1])
userEps = float(sys.argv[2])
datasetName = sys.argv[3]
subsampleSize = int(sys.argv[4])


def coresetSample(x, a, y, rval, levScores):
    """
    Randomly subsample a smaller dataset of certain size
    """
    # gives sizes for each group
    print("starting to get the coreset....")
    AkSizes = Counter(a)
    # AkSizes[0] and AkSizes[1]

    lGroups = len(AkSizes)


    d = sum(levScores)
    logl = math.log(lGroups, 2)
    log1ByEta = math.log(1.0/eta, 2)

    tpart = d + logl + log1ByEta
    invAkSizes = np.array([1.0/AkSizes[x] for x in a])
    tprob = tpart * invAkSizes

    samplingProb = levScores + tprob

    # rvals = [2**i for i in range(0, int(math.log(x.shape[0], 2)-1))]
    # rvals = [0.5, 1, 1.5, 2.0]

    weights = {}
    sampledX = {}
    sampledA = {}
    sampledY = {}
    uniqueForRVal = {}

    weightsOut = []

    print("Going for sampling...")
    # for j in range(len(rvals)):

    # r = rvals[j]
    r = rval

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
            if _DUPLICATE:
                # add point weight many times
                for w in range(weight_i):
                    sampledX[r] = sampledX[r].append(x.iloc[i])
                    sampledA[r] = sampledA[r].append(pd.Series([a[i]]))
                    sampledY[r] = sampledY[r].append(pd.Series([y[i]]))
            else:
                sampledX[r] = sampledX[r].append(x.iloc[i])
                sampledA[r] = sampledA[r].append(pd.Series([a[i]]))
                sampledY[r] = sampledY[r].append(pd.Series([y[i]]))


    sampledX[r].index = range(len(sampledX[r]))
    sampledA[r].index = range(len(sampledX[r]))
    sampledY[r].index = range(len(sampledX[r]))


    print("num Points Sampled -- for r = ", r, " and uniq num points = ", uniqPointsSampled)
    print("num Points Sampled -- for r = ", r, " and num points = ", len(sampledX[r]))

    uniqueForRVal[r] = uniqPointsSampled

    # for i in weightsOut:
        # print(i)

    return sampledX[r], sampledA[r], sampledY[r], uniqueForRVal[r]


def uniformSample(x, a, y, sampleSize, rval):
    """
    Randomly subsample a smaller dataset of certain size
    """
    numPoints = x.shape[0]

    # gives sizes for each group
    print("starting to get the uniform sample of size ---....", sampleSize)
    AkSizes = Counter(a)
    # AkSizes[0] and AkSizes[1]

    lGroups = len(AkSizes)

    # rvals = [2**i for i in range(0, int(math.log(x.shape[0], 2)-1))]
    # rvals = [0.5, 1.0, 1.5, 2.0]

    weights = {}
    sampledX = {}
    sampledA = {}
    sampledY = {}

    weightsOut = []


    print("Going for sampling...")
    # for j in range(len(rvals)):

    r = rval
    weights[r] = []
    sampledX[r] = pd.DataFrame(columns=x.columns)
    sampledA[r] = pd.Series([])
    sampledY[r] = pd.Series([])

    uniqPointsSampled = 0

    # for i in range(sampleSize):
    uniSampledIds = np.random.choice(range(x.shape[0]), sampleSize)

    for i in range(len(uniSampledIds)):
        sampledId = uniSampledIds[i]

        weight_i = math.floor(numPoints/sampleSize)

        if _DUPLICATE:
            for w in range(weight_i):
                sampledX[r] = sampledX[r].append(x.iloc[sampledId])
                sampledA[r] = sampledA[r].append(pd.Series([a[sampledId]]))
                sampledY[r] = sampledY[r].append(pd.Series([y[sampledId]]))
        else:
            sampledX[r] = sampledX[r].append(x.iloc[sampledId])
            sampledA[r] = sampledA[r].append(pd.Series([a[sampledId]]))
            sampledY[r] = sampledY[r].append(pd.Series([y[sampledId]]))

    sampledX[r].index = range(len(sampledX[r]))
    sampledA[r].index = range(len(sampledX[r]))
    sampledY[r].index = range(len(sampledX[r]))


    # print("num Points Sampled -- for r = ", r, " and uniq num points = ", uniqPointsSampled)
    print("num Points Uniformly Sampled after weighing -- for r = ", r, " and num points = ", len(sampledX[r]))

    # for i in weightsOut:
        # print(i)

    return sampledX[r], sampledA[r], sampledY[r]


def train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED):
    """Split the input dataset into train and test sets

    TODO: Need to make sure both train and test sets have enough
    observations from each subgroup
    """
    # size of the training data
    groups = list(a.unique())
    x_train_sets = {}
    x_test_sets = {}
    y_train_sets = {}
    y_test_sets = {}
    a_train_sets = {}
    a_test_sets = {}

    for g in groups:
        x_g = x[a == g]
        a_g = a[a == g]
        y_g = y[a == g]
        x_train_sets[g], x_test_sets[g], a_train_sets[g], a_test_sets[g], y_train_sets[g], y_test_sets[g] = train_test_split(x_g, a_g, y_g, test_size=TEST_SIZE, random_state=random_seed)

    x_train = pd.concat(x_train_sets.values())
    x_test = pd.concat(x_test_sets.values())
    y_train = pd.concat(y_train_sets.values())
    y_test = pd.concat(y_test_sets.values())
    a_train = pd.concat(a_train_sets.values())
    a_test = pd.concat(a_test_sets.values())

    # resetting the index
    x_train.index = range(len(x_train))
    y_train.index = range(len(y_train))
    a_train.index = range(len(a_train))
    x_test.index = range(len(x_test))
    y_test.index = range(len(y_test))
    a_test.index = range(len(a_test))

    return x_train, a_train, y_train, x_test, a_test, y_test



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


def fair_train_test(dataset, size, eps_list, learner, rval, constraint="DP",
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

    if _TEST:
        x_train, a_train, y_train, x_test, a_test, y_test = train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)
    else:
        x_train, a_train, y_train = x, a, y
    
    x_train.index = range(len(x_train))
    a_train.index = range(len(a_train))
    y_train.index = range(len(y_train))

    for eps in eps_list:

        fair_train_model = {}
        train_evaluation = {}
        if _TEST:
            train_on_test_evaluation = {}


        # ------------------------------------------------------------------------------------------
        print("Running on full original data....")

        fair_train_model[eps] = fairlearn.train_FairRegression(x_train, a_train, y_train, eps, Theta, learner, constraint, loss, init_cache=init_cache)

        train_evaluation[eps] = evaluate.evaluate_FairModel(x_train, a_train, y_train, loss, fair_train_model[eps]['exp_grad_result'], Theta)

        if _TEST:
            train_on_test_evaluation[eps] = evaluate.evaluate_FairModel(x_test, a_test, y_test, loss, fair_train_model[eps]['exp_grad_result'], Theta)

        print("Done for full original data.... for eps = ", eps, "\n\n")
        # ------------------------------------------------------------------------------------------

        if _CORESET:
            print("computing svd now...")
            # svd to get leverage scores
            u,s,v = np.linalg.svd(x_train, full_matrices=False)
            sqrtLevScores = np.linalg.norm(u, 2, 1)
            levScores = np.square(sqrtLevScores)
            print("Got the svd...")


        for runId in range(1,4):

            print("RunId = ", runId, " For Epsilon = ", eps, "\n\n")

            result = {}

            fair_cor_model = {}
            cor_evaluation = {}
            cor_on_train_evaluation = {}
            train_on_cor_evaluation = {}

            fair_uni_model = {}
            uni_evaluation = {}
            uni_on_train_evaluation = {}
            train_on_uni_evaluation = {}

            if _TEST:
                cor_on_test_evaluation = {}
                uni_on_test_evaluation = {}


            if _CORESET:

                xcor, acor, ycor, uniqSamples = coresetSample(x_train, a_train, y_train, rval, levScores)
                print("got the coreset")


                # ------------------------------------------------------------------------------------------

                print("Running on the coreset....")
                fair_cor_model[eps] = fairlearn.train_FairRegression(xcor, acor, ycor, eps, Theta, learner, constraint, loss, init_cache=init_cache)

                print("Evaluating Coreset model on Coreset Sketch...")
                cor_evaluation[eps] = evaluate.evaluate_FairModel(xcor, acor, ycor, loss, fair_cor_model[eps]['exp_grad_result'], Theta)

                print("Applying Coreset model on Train Data")
                cor_on_train_evaluation[eps] = evaluate.evaluate_FairModel(x_train, a_train, y_train, loss, fair_cor_model[eps]['exp_grad_result'], Theta)
                
                if _TEST:
                    print("Applying Coreset model on Test Data")
                    cor_on_test_evaluation[eps] = evaluate.evaluate_FairModel(x_test, a_test, y_test, loss, fair_cor_model[eps]['exp_grad_result'], Theta)

                print("Applying Train model on Coreset")
                train_on_cor_evaluation[eps] = evaluate.evaluate_FairModel(xcor, acor, ycor, loss, fair_train_model[eps]['exp_grad_result'], Theta)




            if _UNIFORM:

                xuni, auni, yuni = uniformSample(x_train, a_train, y_train, uniqSamples, rval)
                print("got the unformly sampled points...")
                

                # ------------------------------------------------------------------------------------------
                
                print("Running on Uniform samples...")
                fair_uni_model[eps] = fairlearn.train_FairRegression(xuni, auni, yuni, eps, Theta, learner, constraint, loss, init_cache=init_cache)
                
                print("Evaluating Uniform model on Uni Sketch...")
                uni_evaluation[eps] = evaluate.evaluate_FairModel(xuni, auni, yuni, loss, fair_uni_model[eps]['exp_grad_result'], Theta)

                print("Applying Uni model on Train Data")
                uni_on_train_evaluation[eps] = evaluate.evaluate_FairModel(x_train, a_train, y_train, loss, fair_uni_model[eps]['exp_grad_result'], Theta)

                if _TEST:
                    print("Applying Uni model on Test Data")
                    uni_on_test_evaluation[eps] = evaluate.evaluate_FairModel(x_test, a_test, y_test, loss, fair_uni_model[eps]['exp_grad_result'], Theta)

                print("Applying train model on Uniform Sketch")
                train_on_uni_evaluation[eps] = evaluate.evaluate_FairModel(xuni, auni, yuni, loss, fair_train_model[eps]['exp_grad_result'], Theta)

                # ------------------------------------------------------------------------------------------
        
            

            result['dataset'] = dataset
            result['learner'] = learner.name
            result['loss'] = loss
            result['constraint'] = constraint
            
            result['train_eval'] = train_evaluation
            if _TEST:
                result['train_on_test_eval'] = train_on_test_evaluation
            

            result['cor_eval'] = cor_evaluation
            result['cor_on_train_eval'] = cor_on_train_evaluation
            result['train_on_cor_eval'] = train_on_cor_evaluation
            if _TEST:
                result['cor_on_test_eval'] = cor_on_test_evaluation


            result['uni_eval'] = uni_evaluation
            result['uni_on_train_eval'] = uni_on_train_evaluation
            result['train_on_uni_eval'] = train_on_uni_evaluation
            if _TEST:
                result['uni_on_test_eval'] = uni_on_test_evaluation


            print("\n\n\n")
            print("RunId = ", runId, " For Epsilon = ", eps)

            read_result_list([result])

            # Saving the result list
            outfile = open(dataset + str(runId) + "-" + str(r) + "-" + str(eps) + '-' + str(size) + '.pkl','wb')
            pickle.dump(result, outfile)
            outfile.close()

        print("done with epsilon", eps)

    return None


def read_result_list(result_list):
    """
    Parse the experiment a list of experiment result and print out info
    """

    for result in result_list:
        learner = result['learner']
        dataset = result['dataset']

        train_eval = result['train_eval']
        cor_eval = result['cor_eval']
        uni_eval = result['uni_eval']
        
        cor_on_train_eval = result['cor_on_train_eval']
        uni_on_train_eval = result['uni_on_train_eval']

        train_on_cor_eval = result['train_on_cor_eval']
        train_on_uni_eval = result['train_on_uni_eval']

        if _TEST:
            train_on_test_eval = result['train_on_test_eval']
            cor_on_test_eval = result['cor_on_test_eval']
            uni_on_test_eval = result['uni_on_test_eval']
        
        loss = result['loss']
        constraint = result['constraint']
        learner = result['learner']
        dataset = result['dataset']
        eps_vals = train_eval.keys()
        
        train_disp_dic = {}
        cor_disp_dic = {}
        uni_disp_dic = {}

        cor_on_train_disp_dic = {}
        uni_on_train_disp_dic = {}
        
        train_on_cor_disp_dic = {}
        train_on_uni_disp_dic = {}

        train_err_dic = {}
        cor_err_dic = {}
        uni_err_dic = {}

        cor_on_train_err_dic = {}
        uni_on_train_err_dic = {}
        
        train_on_cor_err_dic = {}
        train_on_uni_err_dic = {}
        
        if _TEST:
            train_on_test_disp_dic = {}
            train_on_test_err_dic = {}

            cor_on_test_disp_dic = {}
            cor_on_test_err_dic = {}

            uni_on_test_disp_dic = {}
            uni_on_test_err_dic = {}

            test_loss_std_dic = {}
            train_on_test_disp_dev_dic = {}

        for eps in eps_vals:
            train_disp = train_eval[eps]["DP_disp"]
            cor_disp = cor_eval[eps]["DP_disp"]
            uni_disp = uni_eval[eps]["DP_disp"]
            
            cor_on_train_disp = cor_on_train_eval[eps]["DP_disp"]
            uni_on_train_disp = uni_on_train_eval[eps]["DP_disp"]

            train_on_cor_disp = train_on_cor_eval[eps]["DP_disp"]
            train_on_uni_disp = train_on_uni_eval[eps]["DP_disp"]

            if _TEST:
                train_on_test_disp = train_on_test_eval[eps]["DP_disp"]
                cor_on_test_disp = cor_on_test_eval[eps]["DP_disp"]
                uni_on_test_disp = uni_on_test_eval[eps]["DP_disp"]

            train_disp_dic[eps] = train_disp
            cor_disp_dic[eps] = cor_disp
            uni_disp_dic[eps] = uni_disp

            cor_on_train_disp_dic[eps] = cor_on_train_disp
            train_on_cor_disp_dic[eps] = train_on_cor_disp

            uni_on_train_disp_dic[eps] = uni_on_train_disp
            train_on_uni_disp_dic[eps] = train_on_uni_disp

            if _TEST:
                train_on_test_disp_dic[eps] = train_on_test_disp
                cor_on_test_disp_dic[eps] = cor_on_test_disp
                uni_on_test_disp_dic[eps] = uni_on_test_disp

            # test_loss_std_dic[eps] = train_on_test_eval[eps]['loss_std']
            # train_on_test_disp_dev_dic[eps] = train_on_test_eval[eps]['disp_std']

            if loss == "square":
                # taking the RMSE
                train_err_dic[eps] = np.sqrt(train_eval[eps]['weighted_loss'])
                cor_err_dic[eps] = np.sqrt(cor_eval[eps]['weighted_loss'])
                uni_err_dic[eps] = np.sqrt(uni_eval[eps]['weighted_loss'])

                cor_on_train_err_dic[eps] = np.sqrt(cor_on_train_eval[eps]['weighted_loss'])
                train_on_cor_err_dic[eps] = np.sqrt(train_on_cor_eval[eps]['weighted_loss'])

                uni_on_train_err_dic[eps] = np.sqrt(uni_on_train_eval[eps]['weighted_loss'])
                train_on_uni_err_dic[eps] = np.sqrt(train_on_uni_eval[eps]['weighted_loss'])

                if _TEST:
                    train_on_test_err_dic[eps] = np.sqrt(train_on_test_eval[eps]['weighted_loss'])
                    cor_on_test_err_dic[eps] = np.sqrt(cor_on_test_eval[eps]['weighted_loss'])
                    uni_on_test_err_dic[eps] = np.sqrt(uni_on_test_eval[eps]['weighted_loss'])

            else:
                train_err_dic[eps] = (train_eval[eps]['weighted_loss'])
                cor_err_dic[eps] = (cor_eval[eps]['weighted_loss'])
                uni_err_dic[eps] = (uni_eval[eps]['weighted_loss'])

                cor_on_train_err_dic[eps] = (cor_on_train_eval[eps]['weighted_loss'])
                train_on_cor_err_dic[eps] = (train_on_cor_eval[eps]['weighted_loss'])

                uni_on_train_err_dic[eps] = (uni_on_train_eval[eps]['weighted_loss'])
                train_on_uni_err_dic[eps] = (train_on_uni_eval[eps]['weighted_loss'])
                
                if _TEST:
                    train_on_test_err_dic[eps] = (train_on_test_eval[eps]['weighted_loss'])
                    cor_on_test_err_dic[eps] = (cor_on_test_eval[eps]['weighted_loss'])
                    uni_on_test_err_dic[eps] = (uni_on_test_eval[eps]['weighted_loss'])

        # taking the pareto frontier
        train_disp_list = [train_disp_dic[k] for k in eps_vals]
        cor_disp_list = [cor_disp_dic[k] for k in eps_vals]
        uni_disp_list = [uni_disp_dic[k] for k in eps_vals]

        cor_on_train_disp_list = [cor_on_train_disp_dic[k] for k in eps_vals]
        train_on_cor_disp_list = [train_on_cor_disp_dic[k] for k in eps_vals]

        uni_on_train_disp_list = [uni_on_train_disp_dic[k] for k in eps_vals]
        train_on_uni_disp_list = [train_on_uni_disp_dic[k] for k in eps_vals]

        if _TEST:
            train_on_test_disp_list = [train_on_test_disp_dic[k] for k in eps_vals]
            cor_on_test_disp_list = [cor_on_test_disp_dic[k] for k in eps_vals]
            uni_on_test_disp_list = [uni_on_test_disp_dic[k] for k in eps_vals]

        train_err_list = [train_err_dic[k] for k in eps_vals]
        cor_err_list = [cor_err_dic[k] for k in eps_vals]
        uni_err_list = [uni_err_dic[k] for k in eps_vals]

        cor_on_train_err_list = [cor_on_train_err_dic[k] for k in eps_vals]
        train_on_cor_err_list = [train_on_cor_err_dic[k] for k in eps_vals]

        uni_on_train_err_list = [uni_on_train_err_dic[k] for k in eps_vals]
        train_on_uni_err_list = [train_on_uni_err_dic[k] for k in eps_vals]

        if _TEST:
            train_on_test_err_list = [train_on_test_err_dic[k] for k in eps_vals]
            cor_on_test_err_list = [cor_on_test_err_dic[k] for k in eps_vals]
            uni_on_test_err_list = [uni_on_test_err_dic[k] for k in eps_vals]

        if loss == "square":
            show_loss = 'RMSE'
        else:
            show_loss = loss


        info = str('Dataset: '+ dataset + '; loss: ' + loss + '; Solver: '+ learner)
        print(info)

        train_data = {'specified epsilon': list(eps_vals), 'SP disparity':train_disp_list, show_loss : train_err_list}
        train_performance = pd.DataFrame(data=train_data)

        cor_data = {'specified epsilon': list(eps_vals), 'SP disparity':cor_disp_list, show_loss : cor_err_list}
        cor_performance = pd.DataFrame(data=cor_data)

        uni_data = {'specified epsilon': list(eps_vals), 'SP disparity':uni_disp_list, show_loss : uni_err_list}
        uni_performance = pd.DataFrame(data=uni_data)

        cor_on_train_data = {'specified epsilon': list(eps_vals), 'SP disparity':cor_on_train_disp_list, show_loss : cor_on_train_err_list}
        cor_on_train_performance = pd.DataFrame(data=cor_on_train_data)  

        train_on_cor_data = {'specified epsilon': list(eps_vals), 'SP disparity':train_on_cor_disp_list, show_loss : train_on_cor_err_list}
        train_on_cor_performance = pd.DataFrame(data=train_on_cor_data)

        uni_on_train_data = {'specified epsilon': list(eps_vals), 'SP disparity':uni_on_train_disp_list, show_loss : uni_on_train_err_list}
        uni_on_train_performance = pd.DataFrame(data=uni_on_train_data)  

        train_on_uni_data = {'specified epsilon': list(eps_vals), 'SP disparity':train_on_uni_disp_list, show_loss : train_on_uni_err_list}
        train_on_uni_performance = pd.DataFrame(data=train_on_uni_data)

        if _TEST:
            train_on_test_data = {'specified epsilon': list(eps_vals), 'SP disparity':train_on_test_disp_list, show_loss : train_on_test_err_list}
            train_on_test_performance = pd.DataFrame(data=train_on_test_data)

            cor_on_test_data = {'specified epsilon': list(eps_vals), 'SP disparity':cor_on_test_disp_list, show_loss : cor_on_test_err_list}
            cor_on_test_performance = pd.DataFrame(data=cor_on_test_data)

            uni_on_test_data = {'specified epsilon': list(eps_vals), 'SP disparity':uni_on_test_disp_list, show_loss : uni_on_test_err_list}
            uni_on_test_performance = pd.DataFrame(data=uni_on_test_data)



        # Print out experiment info.
        print('Train set trade-off:')
        print(train_performance)
        print("------------------------------------------------")
        print('Coreset set trade-off:')
        print(cor_performance)
        print("------------------------------------------------")
        print('Uniform Sketch set trade-off:')
        print(uni_performance)
        print("------------------------------------------------")

        print('Coreset Model on Train set trade-off:')
        print(cor_on_train_performance)
        print("------------------------------------------------")
        print('Uniform Model on Train set trade-off:')
        print(uni_on_train_performance)
        print("------------------------------------------------")

        if _TEST:
            print('Train on Test set trade-off:')
            print(train_on_test_performance)
            print("------------------------------------------------")
            
            print('Cor on Test set trade-off:')
            print(cor_on_test_performance)
            print("------------------------------------------------")
            
            print('Uni on Test set trade-off:')
            print(uni_on_test_performance)
            print("------------------------------------------------")


        print('Train Model on Coreset set trade-off:')
        print(train_on_cor_performance)
        print("------------------------------------------------")
        print('Train on Uniform data set trade-off:')
        print(train_on_uni_performance)
        print("------------------------------------------------")



# Sample instantiation of running the fair regeression algorithm
# eps_list = [0.275, 0.31, 1] # range of specified disparity values
eps_list = [userEps] # range of specified disparity values

# n = 5000  # size of the sub-sampled dataset, when the flag SMALL is True
n = subsampleSize  # size of the sub-sampled dataset, when the flag SMALL is True
# dataset = "adult"  # name of the data set
dataset = datasetName  # name of the data set
constraint = "DP"  # name of the constraint; so far limited to demographic parity (or statistical parity)
loss = "square"  # name of the loss function
learner = solvers.LeastSquaresLearner(Theta) # Specify a supervised learning oracle oracle 

info = str('Dataset: '+dataset + '; loss: ' + loss + '; eps list: '+str(eps_list)) + '; Solver: '+learner.name
print('Starting experiment. ' + info)

# rvals = [0.5, 1, 1.5, 2]
# allResults = []

r = userR
# for r in rvals:
# Run the fair learning algorithm the supervised learning oracle
result = fair_train_test(dataset, n, eps_list, learner, r, 
                          constraint=constraint, loss=loss,
                          random_seed=DATA_SPLIT_SEED)

'''
# read_result_list([result])  # A simple print out for the experiment

# print("Done for r = ", r)

# Saving the result list
outfile = open(info + str(r) +'.pkl','wb')
pickle.dump(result, outfile)
outfile.close()
'''



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

