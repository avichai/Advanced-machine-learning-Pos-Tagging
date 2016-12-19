from pos_tag import pos_tagging
from pos_tag import parser
from pos_tag import phi_models
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os
import pickle
#TODO -link y axis
# TODO - add legends to axes

class TestingResult:
    def __init__(self, TRAIN_DATA_PERCNTAGES, NUM_REPITIONS):
        self.TRAIN_DATA_PERCNTAGES = TRAIN_DATA_PERCNTAGES
        self.NUM_REPITIONS = NUM_REPITIONS

        self.results_time = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_train_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_test_LL = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_sampled_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

        self.results_perceptron_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_perceptron_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

        self.results_perceptron_char_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_perceptron_char_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

        self.results_perceptron_complex_train_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))
        self.results_perceptron_complex_test_err = np.zeros((len(TRAIN_DATA_PERCNTAGES), NUM_REPITIONS))

    def dump(self, pth):
        pdir, name = os.path.split(pth)
        if not os.path.exists(pdir):
            os.mkdir(pdir)
        with open(pth, 'wb') as fid:
            pickle.dump(self, fid)


    def plot_results(self, title, mat, ax, side='right'):
        """
        Plot specific result on a subplot
        :param title: subplot title
        :param mat: The matrix with the results - each row will be plotted as a single line
        """
        ax.title.set_text(title)
        for i in range(len(self.TRAIN_DATA_PERCNTAGES)):
            ax.plot(range(1, self.NUM_REPITIONS + 1), mat[i, :], '-*', label='SampleSize:{0}%'.format(self.TRAIN_DATA_PERCNTAGES[i]*100))
            ax.hold(True)
        if side =='left':
            ax.legend(loc=1, bbox_to_anchor=(-0.05, 1))
        else:
            ax.legend(loc=2, bbox_to_anchor=(1, 1))
        ax.set_xlim([0.9, self.NUM_REPITIONS+0.1])
        ax.xaxis.set_ticks(np.arange(1, self.NUM_REPITIONS + 1, 1, dtype=np.int32))

    def plot_initial_plots(self):
        f = plt.figure()
        f.suptitle("Performance as a function of training data size")
        ax = plt.subplot(3, 2, 1)
        self.plot_results('Sample Number Vs. Train Time (Per sample size)', self.results_time, ax, 'left')
        ax = plt.subplot(3, 2, 3)
        self.plot_results('Sample Number Vs. Train LogLikelihood (Per sample size)', self.results_train_LL, ax, 'left')
        ax = plt.subplot(3, 2, 5)
        self.plot_results('Sample Number Vs. Test LogLikelihood (Per sample size)', self.results_test_LL, ax, 'left')
        ax = plt.subplot(3, 2, 2)
        self.plot_results('Sample Number Vs. Sampled Error (Per sample size)', self.results_sampled_err, ax)
        ax = plt.subplot(3, 2, 4)
        self.plot_results('Sample Number Vs. Train Error (Per sample size)', self.results_train_err, ax)
        ax = plt.subplot(3, 2, 6)
        self.plot_results('Sample Number Vs. Test Error (Per sample size)', self.results_test_err, ax)

    def plot_perceptron_plots(self):
        f = plt.figure()
        f.suptitle("Perceptron plots")
        ax = plt.subplot(3, 2, 1)
        self.plot_results('Simple perc train', self.results_perceptron_train_err, ax, 'left')
        ax = plt.subplot(3, 2, 2)
        self.plot_results('Simple perc test', self.results_perceptron_test_err, ax)
        ax = plt.subplot(3, 2, 3)
        self.plot_results('char perc train', self.results_perceptron_char_train_err, ax, 'left')
        ax = plt.subplot(3, 2, 4)
        self.plot_results('char perc test', self.results_perceptron_char_test_err, ax)
        ax = plt.subplot(3, 2, 5)
        self.plot_results('complex perc train', self.results_perceptron_complex_train_err, ax, 'left')
        ax = plt.subplot(3, 2, 6)
        self.plot_results('complex perc test', self.results_perceptron_complex_test_err, ax)

    def plot(self):
        self.plot_initial_plots()
        self.plot_perceptron_plots()
        plt.show(block=True)



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

    # UT
    for i, w in enumerate(wordslist):
        if wordsDict[w] != i:
            raise Exception("list and dict building was not correct")
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
    NUM_REPITIONS = 2
    TRAIN_DATA_PERCNTAGES = [0.1, 0.25, 0.5, 0.9]
    zippth = './data_split.gz'
    pickle_savepth = './results.pickle'
    SAMPLE_SIZE_FOR_ERROR = 1000  # TODO - if 0 then on everything
    RATE = 0.03


    # init results structs
    results = TestingResult(TRAIN_DATA_PERCNTAGES, NUM_REPITIONS)

    data, xv, yv = parser.collect_sets(zippth, k=10, n=NUM_REPITIONS)  # 9/10 of data is for training, only one copy

    xvDict, xvlist = set2DictAndList(xv)
    yvDict, yvlist = set2DictAndList(yv)

    # Test
    for pidx, perc in enumerate(TRAIN_DATA_PERCNTAGES):
        if perc > 0.9:
            raise Exception("Percentage is to big")

        for rep in range(NUM_REPITIONS):
            print("-----------------Starting perc: {0}/{1}, rep: {2}/{3}----------------".format(pidx+1, len(TRAIN_DATA_PERCNTAGES),rep+1,NUM_REPITIONS))

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
            print('train')
            t_hat, e_hat, q_hat, results.results_train_LL[pidx, rep] = pos_tagging.mle(trainX, trainY, xvDict, yvDict)
            results.results_time[pidx, rep] = time() - start_time
            ni, nij, nyi = pos_tagging.get_ni_nij_nyi(X_test, Y_test, xvDict, yvDict)
            print('test LL')
            results.results_test_LL[pidx, rep] = pos_tagging.log_likelihood(q_hat, t_hat, e_hat, ni, nij, nyi)

            # Sample data using MLE
            print('Sample')
            curr_sample_size_for_error = int(perc*SAMPLE_SIZE_FOR_ERROR)
            sentencesx, sentencesy = pos_tagging.sample(np.random.randint(low=7, high=35, size=curr_sample_size_for_error),
                                                        xvlist, yvlist, t_hat, e_hat, q_hat)

            # Test inference (Against sampled data, train data, test data)
            simple_phi, D = phi_models.get_hmm_phi(xvDict, yvDict)
            simple_w = phi_models.get_hmm_w_vec(t_hat, e_hat, q_hat, xvDict, yvDict)

            train_rnd_ind = np.random.choice(a=np.arange(len(trainX)), size=curr_sample_size_for_error)
            test_rnd_ind = np.random.choice(a=np.arange(len(X_test)), size=curr_sample_size_for_error)



            #### starting perceptron checks: ###
            if perc == 0.1:
                w0 = np.zeros(D)
                print("Perceptron simple phi space")
                w_hat = pos_tagging.perceptron(trainX, trainY, xvlist, simple_phi, w0, RATE) #todo check different rates

                char_phi, D2 = phi_models.get_word_carachteristics_phi()
                phi_complex, D3 = phi_models.get_complex_phi(char_phi, D2, simple_phi, D)

                w0_char = np.zeros(D2)
                w0_complex = np.zeros(D3)

                print("Perceptron simple char space")
                w_char = pos_tagging.perceptron(trainX, trainY, xvlist, char_phi, w0_char, RATE) #todo check different rates
                print("Perceptron simple complex space")
                w_complex = pos_tagging.perceptron(trainX, trainY, xvlist, phi_complex, w0_complex, RATE) #todo check different rates


                print('Inf error sample')
                results.results_sampled_err[pidx, rep] = get_inference_err(sentencesx, sentencesy, xvlist,                                                               simple_phi, simple_w)
                print('Inf error train')
                results.results_train_err[pidx, rep] = get_inference_err(trainX[train_rnd_ind],
                                                                         trainY[train_rnd_ind], xvlist,
                                                                         simple_phi, simple_w)
                print('Inf error test')
                results.results_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind], testY[test_rnd_ind],
                                                                        xvlist, simple_phi, simple_w)
                print('Inf simple perceptron error train')
                results.results_perceptron_train_err[pidx, rep] = get_inference_err(trainX[train_rnd_ind],
                                                                                    trainY[train_rnd_ind], xvlist,
                                                                                        simple_phi, w_hat)
                print('Inf simple perceptron error test')
                results.results_perceptron_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind],
                                                                                   testY[test_rnd_ind], xvlist,
                                                                                   simple_phi, w_hat)
                print('Inf char perceptron error train')
                results.results_perceptron_char_train_err[pidx, rep] = get_inference_err(trainX[train_rnd_ind],
                                                                                         trainY[train_rnd_ind],
                                                                                         xvlist,
                                                                                         char_phi, w_char)
                print('Inf char perceptron error test')
                results.results_perceptron_char_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind],
                                                                                        testY[test_rnd_ind],
                                                                                        xvlist,
                                                                                        char_phi, w_char)
                print('Inf complex perceptron error train')
                results.results_perceptron_complex_train_err[pidx, rep] = get_inference_err(trainX[train_rnd_ind],
                                                                                            trainY[train_rnd_ind],
                                                                                            xvlist,
                                                                                            phi_complex,
                                                                                            w_complex)
                print('Inf complex perceptron error test')
                results.results_perceptron_complex_test_err[pidx, rep] = get_inference_err(testX[test_rnd_ind],
                                                                                           testY[test_rnd_ind],
                                                                                           xvlist,
                                                                                           phi_complex, w_complex)

    results.dump(pickle_savepth)
    results.plot()
    return results


if __name__ == '__main__':
    main()
