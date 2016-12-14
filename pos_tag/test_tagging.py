from pos_tag import pos_tagging
from pos_tag import parser
import pickle
import os

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

def test_mle():
    sample = pickle.load(open('sample.pickle','rb'))
    for perc in [0.01, 0.1, 0.25, 0.5, 0.9]:
        t_hat, e_hat, q_hat = pos_tagging.mle(sample.X[:int(len(sample.X) * perc)], sample.Y[:int(len(sample.Y) * perc)],
                                       sample.xv, sample.yv)
        sentencesx, sentencesy = pos_tagging.sample([10, 8, 15, 7,10,4,8,20,25], sample.xv, sample.yv, t_hat, e_hat, q_hat)
        # for x,y in zip(sentencesx, sentencesy):
        #     print("X:{0}\nY:{1}".format(x,[word for word in y]))

        for i in range(len(sentencesx)):
            sentencesx_hat = pos_tagging.viterbi(sentencesy[i], sample.xv, sample.yv, t_hat, e_hat, q_hat)
            print('Expected: {0},\nActual:   {1}\n'.format(sentencesx[i], sentencesx_hat.tolist()))



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