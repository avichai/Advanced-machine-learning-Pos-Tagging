from pos_tag import pos_tagging
from pos_tag import parser
from pos_tag import phi_models
import numpy as np
from time import time
import matplotlib.pyplot as plt

def set2DictAndList(words_set):
    wordsDict = {x: i for i, x in enumerate(words_set)}
    wordslist = [''] * len(wordsDict)
    for k, v in wordsDict.items():
        wordslist[v] = k
    return wordsDict, wordslist

def get_inference_err(sentencesx, sentencesy, xvlist, phi, w):
    correct = 0
    total_wrods = 0
    for sentx, senty in zip(sentencesx, sentencesy):
        sentencesx_hat = pos_tagging.viterbi(senty, xvlist, phi, w)
        total_wrods += len(sentx)
        correct += np.count_nonzero(np.asarray(sentx) == np.asarray(sentencesx_hat))
    return 1 - correct / float(total_wrods)

def main():
    """
    This function is designed to test the MLE. In order to do so TODO
    :return:
    """
    NUM_REPITIONS = 2
    TRAIN_DATA_PERCNTAGES = [0.1, 0.2] #[0.1, 0.25, 0.5, 0.9]
    zippth = '../data_split.gz'
    SAMPLE_SIZE_FOR_ERROR = 5

    # init results structs
    results_time = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_train_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_test_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_sampled_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

    data, xv, yv = parser.collect_sets(zippth, k=10, n=NUM_REPITIONS)  # 9/10 of data is for training, only one copy

    xvDict, xvlist = set2DictAndList(xv)
    yvDict, yvlist = set2DictAndList(yv)

    # Test
    for pidx, perc in enumerate(TRAIN_DATA_PERCNTAGES):
        if perc > 0.9:
            raise Exception("Percentage is to big")

        for rep in range(NUM_REPITIONS):
            # Sampling data
            X_train = [d[0] for d in data[rep]['train']]
            Y_train = [d[1] for d in data[rep]['train']]
            X_test = [d[0] for d in data[rep]['test']]
            Y_test = [d[1] for d in data[rep]['test']]
            trainX = np.asarray(X_train)[:int(len(X_train)*perc*10.0/9)]
            trainY = np.asarray(Y_train)[:int(len(Y_train)*perc*10.0/9)]
            testX = np.asarray(X_test)
            testY = np.asarray(Y_test)

            # Estimate ML and get LL
            start_time = time()
            t_hat, e_hat, q_hat, results_train_LL[pidx, rep] = pos_tagging.mle(trainX, trainY, xvDict, yvDict)
            results_time[pidx, rep] = time() - start_time
            ni, nij, nyi = pos_tagging.get_ni_nij_nyi(X_test, Y_test, xvDict, yvDict)
            results_test_LL[pidx, rep] = pos_tagging.log_likelihood(q_hat, t_hat, e_hat, ni, nij, nyi)

            # TODO - remove
            print(results_time[pidx, rep], results_train_LL[pidx, rep], results_test_LL[pidx, rep])


            # Sample data using MLE
            sentencesx, sentencesy = pos_tagging.sample(np.random.randint(low=4, high=40, size=SAMPLE_SIZE_FOR_ERROR),
                                                        xvlist, yvlist, t_hat, e_hat, q_hat)

            # Test inference (Against sampled data, train data, test data)
            simple_phi, D = phi_models.get_hmm_phi(xvDict, yvDict)
            simple_w = phi_models.get_hmm_w_vec(t_hat, e_hat, q_hat, xvDict, yvDict)

            train_rnd_ind = np.random.choice(a=np.arange(len(trainX)), size=SAMPLE_SIZE_FOR_ERROR)
            test_rnd_ind = np.random.choice(a=np.arange(len(X_test)), size=SAMPLE_SIZE_FOR_ERROR)
            results_sampled_err[pidx, rep] = get_inference_err(sentencesx, sentencesy, xvlist, simple_phi, simple_w)
            results_train_err[pidx, rep] = get_inference_err(trainX[train_rnd_ind], trainY[train_rnd_ind], xvlist, simple_phi, simple_w)
            results_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind], testY[test_rnd_ind], xvlist, simple_phi, simple_w)

    f = plt.figure()
    f.suptitle("Performance as a function of training data size")

    plt.subplot(3,2,1)
    plt.title('Sample Number Vs. Train Time (Per sample size)')
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.plot(range(1, NUM_REPITIONS+1), results_time[i, :], label='SampleSize:{0}%'.format(TRAIN_DATA_PERCNTAGES[i]))
        plt.hold(True)
    plt.legend()

    plt.subplot(3, 2, 3)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.title('Sample Number Vs. Train LogLikelihood (Per sample size)')
        plt.plot(range(1, NUM_REPITIONS+1), results_train_LL[i, :], label='SampleSize:{0}%'.format(TRAIN_DATA_PERCNTAGES[i]))
        plt.hold(True)
    plt.legend()

    plt.subplot(3, 2, 5)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.title('Sample Number Vs. Test LogLikelihood (Per sample size)')
        plt.plot(range(1, NUM_REPITIONS+1), results_test_LL[i, :], label='SampleSize:{0}%'.format(TRAIN_DATA_PERCNTAGES[i]))
        plt.hold(True)
    plt.legend()

    plt.subplot(3, 2, 2)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.title('Sample Number Vs. Sampled Error (Per sample size)')
        plt.plot(range(1, NUM_REPITIONS+1), results_train_err[i, :], label='SampleSize:{0}%'.format(TRAIN_DATA_PERCNTAGES[i]))
        plt.hold(True)
    plt.legend()

    plt.subplot(3, 2, 4)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.title('Sample Number Vs. Train Error (Per sample size)')
        plt.plot(range(1, NUM_REPITIONS+1), results_train_err[i, :], label='SampleSize:{0}%'.format(TRAIN_DATA_PERCNTAGES[i]))
        plt.hold(True)
    plt.legend()

    plt.subplot(3, 2, 6)
    for i in range(len(TRAIN_DATA_PERCNTAGES)):
        plt.title('Sample Number Vs. Test Error (Per sample size)')
        plt.plot(range(1, NUM_REPITIONS+1), results_test_err[i, :], label='SampleSize:{0}%'.format(TRAIN_DATA_PERCNTAGES[i]))
        plt.hold(True)
    plt.legend()

    plt.show(block=True)


if __name__ == '__main__':
    main()
