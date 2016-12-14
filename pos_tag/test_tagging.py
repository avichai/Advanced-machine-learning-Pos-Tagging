from pos_tag import pos_tagging
from pos_tag import parser
import pickle
import os
import numpy as np

class Sample:
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
            pickle.dump(Sample(X, Y, xv, yv, X_test, Y_test), f)

def get_simple_phi_and_w(t, e, q, suppX, suppY):
    S, T = len(suppX), len(suppY)
    D = S+S*S+T*S
    w = np.zeros(D)
    EPS = 1E-50
    w[:S] = np.log(q+EPS)
    w[S:S+S*S] = np.log(t.flatten()+EPS)
    w[S+S*S:] = np.log(e.flatten()+EPS)

    def simple_phi(xt, xprev, y, t):
        '''
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
        '''
        feature_vec = np.zeros(D)
        xt_idx = np.where(suppX == xt)[0][0]
        xprev_idx = np.where(suppX == xprev)[0][0]
        yidx = np.where(suppY == y[t])[0][0]
        if t == 0:
            feature_vec[xt_idx] = 1
        else:
            feature_vec[S + xprev_idx*S + xt_idx] = 1
        feature_vec[S+S*S+yidx*S+xt_idx] = 1

    return simple_phi, w



def test_mle():
    sample = pickle.load(open('sample.pickle','rb'))
    for perc in [0.01, 0.1, 0.25, 0.5, 0.9]:
        t_hat, e_hat, q_hat = pos_tagging.mle(sample.X[:int(len(sample.X) * perc)], sample.Y[:int(len(sample.Y) * perc)],
                                       sample.xv, sample.yv)
        sentencesx, sentencesy = pos_tagging.sample([10, 8, 15, 7,10,4,8,20,25], sample.xv, sample.yv, t_hat, e_hat, q_hat)
        # for x,y in zip(sentencesx, sentencesy):
        #     print("X:{0}\nY:{1}".format(x,[word for word in y]))
        simple_phi, simple_w = get_simple_phi_and_w(t_hat, e_hat, q_hat, sample.xv, sample.yv)
        for i in range(len(sentencesx)):
            sentencesx_hat = pos_tagging.viterbi_non_general(sentencesy[i], sample.xv, sample.yv, t_hat, e_hat, q_hat)
            sentencesx_hat2 = pos_tagging.viterbi(sentencesy[i], sample.xv, sample.yv, simple_phi, simple_w)
            print('Expected: {0},\nNGV:   {1}\nGV:      {2}\n'.format(sentencesx[i], sentencesx_hat.tolist()), sentencesx_hat2.tolist())



def main():
    try:
        if not os.path.exists('sample.pickle'):
            Sample.generatePickle('data_split.gz', 'sample.pickle')
        for test in [test_mle]:
            test()
    except Exception as e:
        print("Failed test due to : {0}".format(e))
        exit(-1)

if __name__ == '__main__':
    main()