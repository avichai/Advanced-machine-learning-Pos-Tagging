import numpy as np

def mle (x , y):
    '''
        Calculate the maximum likelihood estimators for the transition and
        emission distributions , in the multinomial HMM case .
        : param x : an iterable over sequences of POS tags
        : param : a matching iterable over sequences of words
        : return : a     tuple (t , e ) , with
        t . shape = (| val ( X )| ,| val ( X )|) , and
        e . shape = (| val ( X )| ,| val ( Y )|)
    '''
    S = np.unique(x)
    T = np.unique(y)

    nij = get_nij(x, S)
    nyi = get_nyi(x, y, S, T)
    t_hat = get_estimator(nij)
    e_hat = get_estimator(nyi)
    return t_hat, e_hat


def get_nij(X, S):
    nij = np.zeros(len(S))
    for i, si in enumerate(S):
        for j, sj in enumerate(S):
            nij[i, j] = np.sum(np.logical_and(np.where(X == sj)[1:], np.where(X == si)[:-1]))
    return nij

def get_nyi(X, Y, S, T):
    nyi = np.zeros(len(T), len(S))
    for i, s in enumerate(S):
        for j, t in enumerate(T):
            nyi[i, j] = np.sum(np.logical_and(np.where(X == s), np.where(Y == t)))
    return nyi

def get_estimator(n_mat):
    return np.divide(n_mat, np.matlib.repmat(np.sum(n_mat, 0), 1, n_mat.shape[1]))
