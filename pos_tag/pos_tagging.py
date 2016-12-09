import numpy as np

def mle (x, y, xv, yv):
    '''
        Calculate the maximum likelihood estimators for the transition and
        emission distributions , in the multinomial HMM case .
        : param x : an iterable over sequences of POS tags
        : param : a matching iterable over sequences of words
        : return : a     tuple (t , e ) , with
        t . shape = (| val ( X )| ,| val ( X )|) , and
        e . shape = (| val ( X )| ,| val ( Y )|)
    '''
    nij = get_nij(x, xv)
    nyi = get_nyi(x, y, xv, yv)
    t_hat = get_estimator(nij)
    e_hat = get_estimator(nyi)
    return t_hat, e_hat


def get_nij(X, S):
    # new syntax
    flattened_X = np.asarray([item for sublist in X for item in ['']+sublist+['']])
    occ_mat = np.zeros((len(S), len(flattened_X)))
    for i, s in enumerate(S):
        occ_mat[i, :] = flattened_X == s
    nij = np.matmul(occ_mat[:,:-1], np.transpose(occ_mat[:,1:]))


    nij = np.zeros([len(S)]*2)
    for seq in X:
        seq = np.asarray(seq)
        for i, si in enumerate(S):
            for j, sj in enumerate(S):
                nij[i, j] += np.sum(np.logical_and((seq == sj)[1:], (seq == si)[:-1]))
    return nij

def get_nyi(X, Y, S, T):
    nyi = np.zeros(len(T), len(S))
    for i, s in enumerate(S):
        for j, t in enumerate(T):
            for seqX, seqY in zip(X, Y):
                nyi[i, j] = np.sum(np.logical_and(np.asarray(seqX) == s, np.asarray(seqY) == t))

    return nyi

def get_estimator(n_mat):
    return np.divide(n_mat, np.matlib.repmat(np.sum(n_mat, 0), 1, n_mat.shape[1]))
