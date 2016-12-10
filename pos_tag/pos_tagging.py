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
    occ_mat_X = get_occ_mat(x, xv)
    occ_mat_Y = get_occ_mat(y, yv)
    nij = get_nij(occ_mat_X)
    nyi = get_nyi(occ_mat_X, occ_mat_Y)
    t_hat = get_estimator(nij)
    e_hat = get_estimator(nyi)
    return t_hat, e_hat

def flatten_list(lst):
    return np.asarray([item for sublist in lst for item in ['']+sublist+['']])

def get_occ_mat(lst, vals):
    flat_list = flatten_list(lst)
    occ_mat = np.zeros((len(vals), len(flat_list)))
    for i, v in enumerate(vals):
        occ_mat[i, :] = flat_list == v
    return occ_mat

def get_nij(occ_mat_X):
    return np.matmul(occ_mat_X[:,:-1], np.transpose(occ_mat_X[:,1:]))

def get_nyi(occ_mat_X, occ_mat_Y):
    return np.matmul(occ_mat_Y, occ_mat_X.transpose())

def get_estimator(n_mat):
    return np.divide(n_mat, np.matlib.repmat(np.sum(n_mat, 0), 1, n_mat.shape[1]))
