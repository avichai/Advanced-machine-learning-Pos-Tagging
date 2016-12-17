from pos_tag import pos_tagging
from pos_tag import parser
import pickle
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt

class Corpus:
    def __init__(self, X, Y, xv, yv, X_test, Y_test):
        self.X = X
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y = Y
        self.xv = xv
        self.yv = yv

    @staticmethod
    def generatePickle(zippth, savepth, k=10, n=1):
        data, xv, yv = parser.collect_sets(zippth, k=k, n=n)  # 9/10 of data is for training, only one copy
        X = [d[0] for d in data['train']]
        Y = [d[1] for d in data['train']]
        X_test = [d[0] for d in data['test']]
        Y_test = [d[1] for d in data['test']]
        with open(savepth, 'wb') as f:
            pickle.dump(Corpus(X, Y, xv, yv, X_test, Y_test), f)

def get_simple_phi_and_w(t, e, q, suppX, suppY):
    S, T = len(suppX), len(suppY)
    D = S+S*S+T*S
    w = np.zeros(D)
    EPS = 1E-50
    w[:S] = np.log(q+EPS)
    w[S:S+S*S] = np.log(t.flatten()+EPS)
    w[S+S*S:] = np.log(e.flatten()+EPS)

    def simple_phi(xt, xprev, y, t):
        """
        S is the number of states. T is the number of possible words.
        q is a 1 by S vector, t is a S by S matrix, e is a T by S matrix
        Phi(xt, xprev, y, t) has d entries:
            S entries: if t==0, 1 in idx opf xt, 0 else
            SXS entries: if t !=0 : (1 in s+xprev*S + xt, 0 elsewhere). 0 else
            TXS entries: 1 in index_of(y[t])_in_suppY*S+xt+S+S*S, 0 else
        :param xt:
        :param xprev:
        :param y:
        :param t:
        :return:
        """
        # feature_vec = np.zeros(D)
        indices_vec = [0]*2
        xt_idx = suppX[xt]
        yidx = suppY[y[t]]
        if t == 0:
            indices_vec[0] = xt_idx
            # feature_vec[xt_idx] = 1
        else:
            xprev_idx = suppX[xprev]
            # feature_vec[S + xprev_idx*S + xt_idx] = 1
            indices_vec[0] = S + xprev_idx*S + xt_idx
        indices_vec[1] = S+S*S+yidx*S+xt_idx
        # feature_vec[S+S*S+yidx*S+xt_idx] = 1
        return indices_vec

    return simple_phi, w

def set2DictAndList(words_set):
    wordsDict = {x: i for i, x in enumerate(words_set)}
    wordslist = [''] * len(wordsDict)
    for k, v in wordsDict.items():
        wordslist[v] = k
    return wordsDict, wordslist

def test_mle():
    """
    This function is designed to test the MLE. In order to do so TODO
    :return:
    """
    NUM_REPITIONS = 5
    TRAIN_DATA_PERCNTAGES = [0.1, 0.25, 0.5, 0.9]
    sample = pickle.load(open('sample.pickle','rb'))
    xvDict, xvlist = set2DictAndList(sample.xv)
    yvDict, yvlist = set2DictAndList(sample.yv)

    # init results structs
    results_time = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_train_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_test_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

    # Test
    for pidx, perc in enumerate(TRAIN_DATA_PERCNTAGES):
        if perc > 0.9:
            raise Exception("Percentage is to big")

        for rep in range(NUM_REPITIONS):
            # Sampling data
            indices = np.random.permutation(int(len(sample.X)))
            train_ind, test_ind = indices[:int(len(indices) * (perc))], indices[int(len(indices) *
                                                                                    (perc)):int(len(indices) * (perc+0.1))]
            trainX, trainY = np.asarray(sample.X)[train_ind], np.asarray(sample.Y)[train_ind]
            testX, testY = np.asarray(sample.X)[test_ind], np.asarray(sample.Y)[test_ind]

            start_time = time()
            t_hat, e_hat, q_hat, results_train_LL[pidx, rep] = pos_tagging.mle(trainX, trainY, xvDict, yvDict)
            results_time[pidx, rep] = time() - start_time
            ni, nij, nyi = pos_tagging.get_ni_nij_nyi(testX, testY, xvDict, yvDict)
            results_test_LL[pidx, rep] = pos_tagging.log_likelihood(q_hat, t_hat, e_hat, ni, nij, nyi)

            print(results_time[pidx, rep], results_train_LL[pidx, rep], results_test_LL[pidx, rep])


            sentencesx, sentencesy = pos_tagging.sample([10, 8, 15, 7,10,4,8,20,25], xvlist, yvlist, t_hat, e_hat, q_hat)
            simple_phi, simple_w = get_simple_phi_and_w(t_hat, e_hat, q_hat, xvDict, yvDict)

            t = time()
            # final_w = pos_tagging.perceptron(trainX, trainY, xvlist, simple_phi, np.zeros(len(simple_w)), 0.05)
            print("perceptron took: {0}".format(time() - t))
            for i in range(len(sentencesx)):
                sentencesx_hat = pos_tagging.viterbi_non_general(sentencesy[i], xvlist, yvDict, t_hat, e_hat, q_hat)
                sentencesx_hat2 = pos_tagging.viterbi(sentencesy[i], xvlist, simple_phi, simple_w)
                # sentencesx_hat3 = pos_tagging.viterbi(sentencesy[i], xvlist, simple_phi, final_w)
                print('Expected: {0},\nNGV:      {1}\nGV:       {2}\nPERC:     {3}\n'.format(sentencesx[i], sentencesx_hat.tolist(),
                                                                              sentencesx_hat2.tolist(), []))
            # print("Mean diff between w: {0}".format(np.mean(np.abs(simple_w-final_w))))


    plt.figure()
    plt.subplot(3,1,1)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.plot(results_time[i,:])
        plt.hold(True)
    plt.subplot(3, 1, 2)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.plot(results_train_LL[i, :])
        plt.hold(True)
    plt.subplot(3, 1, 3)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.plot(results_test_LL[i, :])
        plt.hold(True)
    plt.show(block=True)


# sentencesx, sentencesy = pos_tagging.sample([10, 8, 15, 7,10,4,8,20,25], xvlist, yvlist, t_hat, e_hat, q_hat)
#
#
# simple_phi, simple_w = get_simple_phi_and_w(t_hat, e_hat, q_hat, xvDict, yvDict)
#
# t = time()
# final_w = pos_tagging.perceptron(trainX, trainY, xvlist, simple_phi, np.zeros(len(simple_w)), 0.05)
# print("perceptron took: {0}".format(time() - t))
# for i in range(len(sentencesx)):
#     sentencesx_hat = pos_tagging.viterbi_non_general(sentencesy[i], xvlist, yvDict, t_hat, e_hat, q_hat)
#     sentencesx_hat2 = pos_tagging.viterbi(sentencesy[i], xvlist, simple_phi, simple_w)
#     sentencesx_hat3 = pos_tagging.viterbi(sentencesy[i], xvlist, simple_phi, final_w)
#     print('Expected: {0},\nNGV:      {1}\nGV:       {2}\nPERC:     {3}\n'.format(sentencesx[i], sentencesx_hat.tolist(),
#                                                                   sentencesx_hat2.tolist(), sentencesx_hat3.tolist()))
# print("Mean diff between w: {0}".format(np.mean(np.abs(simple_w-final_w))))


def main():
    import traceback
    try:
        if not os.path.exists('sample.pickle'):
            Corpus.generatePickle('data_split.gz', 'sample.pickle')
        for test in [test_mle]:
            test()
    except Exception as e:
        print("Failed test due to : {0}".format(e))
        traceback.print_exc()
        exit(-1)

if __name__ == '__main__':
    main()