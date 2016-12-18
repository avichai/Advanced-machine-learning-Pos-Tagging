from pos_tag import pos_tagging
from pos_tag import parser
from pos_tag import phi_models
import numpy as np
from time import time
import matplotlib.pyplot as plt

def set2DictAndList(words_set):
    """
    create word dictionary and word list.
    :param words_set: a set of words.
    :return: word dictionary and word list.
    """
    wordsDict = {x: i for i, x in enumerate(words_set)}
    wordslist = [''] * len(wordsDict)
    for k, v in wordsDict.items():
        wordslist[v] = k
    return wordsDict, wordslist

def get_inference_err(sentencesx, sentencesy, xvlist, phi, w):
    """
    get the inference error of all sentences that were given.
    :param sentencesx: list of POS divided by sentences
    :param sentencesy: list of words divided by sentences
    :param xvlist: the possible values for x variables , ordered as in t , and e ( Dict - see MLE)
    :param phi: the possible values for y variables , ordered as in e ( Dict - see MLE)
    :param w: the weights.
    :return: the inference error of all sentences that were given.
    """
    correct = 0
    total_wrods = 0
    for sentx, senty in zip(sentencesx, sentencesy):
        sentencesx_hat = pos_tagging.viterbi(senty, xvlist, phi, w)
        total_wrods += len(sentx)
        correct += np.count_nonzero(np.asarray(sentx) == np.asarray(sentencesx_hat))
    return 1 - correct / float(total_wrods)





def main():
    """
    This function is designed to test out functions. In order to do so it runs ML estimation on several chunks
    of data in different sizes, test infernce error (against train data, sampled data and test data).
    Also, it tests different models using the perceptron in prder to learn. Evantually,
    it plots the results for easier viweing.
    """

    # Constants
    NUM_REPITIONS = 2 # todo change
    TRAIN_DATA_PERCNTAGES = [0.1, 0.9] #[0.1, 0.9] #[0.1, 0.25, 0.5, 0.9]
    zippth = './data_split.gz'
    SAMPLE_SIZE_FOR_ERROR = 50

    def plot_results(title, mat, ax):
        """
        Plot specific result on a subplot
        :param title: subplot title
        :param mat: The matrix with the results - each row will be plotted as a single line
        """
        ax.title.set_text(title)
        for i in range(len(TRAIN_DATA_PERCNTAGES)):
            ax.plot(range(1, NUM_REPITIONS + 1), mat[i, :], '-*', label='SampleSize:{0}%'.format(TRAIN_DATA_PERCNTAGES[i]))
            ax.hold(True)
        ax.legend()

    # init results structs
    results_time = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_train_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_test_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_sampled_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

    results_perceptron_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_perceptron_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

    results_perceptron_hmm_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_perceptron_hmm_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

    results_perceptron_char_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_perceptron_char_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

    results_perceptron_complex_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
    results_perceptron_complex_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))


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


            #### starting perceptron checks: ###
            S, T = len(xvDict), len(yvDict)
            D = S + S * S + T * S
            w0 = np.zeros(D)
            w_hat = pos_tagging.perceptron(trainX, trainY, xvlist, simple_phi, w0, 0.05) #todo check different rates
            results_perceptron_train_err[pidx, rep] = get_inference_err(trainX[train_rnd_ind], trainY[train_rnd_ind], xvlist,
                                                             simple_phi, w_hat)
            results_perceptron_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind], testY[test_rnd_ind], xvlist,
                                                            simple_phi, w_hat)

            char_phi, D2 = phi_models.get_word_carachteristics_phi()
            phi_complex, D3 = phi_models.get_complex_phi(char_phi, D2, simple_phi, D)

            w0_char = np.zeros(D2)
            w0_complex = np.zeros(D+D2)

            w_char = pos_tagging.perceptron(trainX, trainY, xvlist, char_phi, w0_char, 0.05) #todo check different rates
            w_complex = pos_tagging.perceptron(trainX, trainY, xvlist, phi_complex, w0_complex, 0.05) #todo check different rates

            results_perceptron_char_train_err[pidx, rep] = get_inference_err(testX[test_rnd_ind],
                                                                                 testY[test_rnd_ind], xvlist,
                                                                             char_phi, w0_char)
            results_perceptron_char_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind],
                                                                                 testY[test_rnd_ind], xvlist,
                                                                            char_phi, w0_char)
            results_perceptron_complex_train_err[pidx, rep] = get_inference_err(testX[test_rnd_ind],
                                                                                 testY[test_rnd_ind], xvlist,
                                                                                phi_complex, w0_complex)
            results_perceptron_complex_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind],
                                                                                 testY[test_rnd_ind], xvlist,
                                                                               phi_complex, w0_complex)

    f = plt.figure()
    f.suptitle("Performance as a function of training data size")
    ax = plt.subplot(3,5,1)
    plot_results('Sample Number Vs. Train Time (Per sample size)', results_time, ax)
    ax = plt.subplot(3,5,6)
    plot_results('Sample Number Vs. Train LogLikelihood (Per sample size)', results_train_LL, ax)
    ax = plt.subplot(3,5,11)
    plot_results('Sample Number Vs. Test LogLikelihood (Per sample size)', results_test_LL, ax)
    ax = plt.subplot(3,5,2)
    plot_results('Sample Number Vs. Sampled Error (Per sample size)', results_sampled_err, ax)
    ax = plt.subplot(3,5,7)
    plot_results('Sample Number Vs. Train Error (Per sample size)', results_train_err, ax)
    ax = plt.subplot(3,5,12)
    plot_results('Sample Number Vs. Test Error (Per sample size)', results_test_err, ax)

    ax = plt.subplot(3, 5, 3)
    plot_results('Sample Number Vs. Perceptron Train Error (Per sample size)', results_perceptron_train_err, ax)
    ax = plt.subplot(3, 5, 8)
    plot_results('Sample Number Vs. Perceptron Test Error (Per sample size)', results_perceptron_test_err, ax)

    ax = plt.subplot(3, 5, 4)
    plot_results('Sample Number Vs. char Train Error (Per sample size)', results_perceptron_char_train_err, ax)
    ax = plt.subplot(3, 5, 9)
    plot_results('Sample Number Vs. char Test Error (Per sample size)', results_perceptron_char_test_err, ax)


    ax = plt.subplot(3, 5, 5)
    plot_results('Sample Number Vs. complex Train Error (Per sample size)', results_perceptron_complex_train_err, ax)
    ax = plt.subplot(3, 5, 10)
    plot_results('Sample Number Vs. complex Test Error (Per sample size)', results_perceptron_complex_test_err, ax)

    plt.show(block=True)


if __name__ == '__main__':
    main()
