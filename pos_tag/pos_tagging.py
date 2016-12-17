import numpy as np
from numpy.matlib import repmat as repmat
from numpy.random import multinomial, choice

MAX_OCC_MAT_SIZE = 0.25e9  # Maximum allowed matrix elements


def mle(x, y, xv, yv):
    """
    Calculate the maximum likelihood estimators for the transition and
    emission distributions , in the multinomial HMM case .
    :param x: an iterable over sequences of POS tags
    :param y: a matching iterable over sequences of words
    :param xv: A dictionary from POS tagging (possible xvalue) to index between 0 to len(xv)-1 - Note that its a
                bijective mapping
    :param yv: A dictionary from Possible word (possible xvalue) to index between 0 to len(yv)-1 - Note that its a
                bijective mapping
    :return: a  tuple (t, e, q), with
            t . shape = (| val ( X )| ,| val ( X )|) , and
            e . shape = (| val ( X )| ,| val ( Y )|)
            q . TODO
    """
    PADDING_CONST = '##$$##'
    # TODO - make sure PADDING_CONST not in xv or yv
    flatten_X, flatten_Y = flatten_list(x, PADDING_CONST), flatten_list(y, PADDING_CONST)
    M = len(flatten_Y)
    S, T = len(xv), len(yv)
    nij = np.zeros((S+1,S+1))
    nyi = np.zeros((T+1,S+1))
    xv[PADDING_CONST] = S
    yv[PADDING_CONST] = T
    for i in range(M-1):
        nij[xv[flatten_X[i]], xv[flatten_X[i+1]]] += 1
        nyi[yv[flatten_Y[i]], xv[flatten_X[i]]] += 1
        # Note that we don't go over the last pair of y and x, but its a padding from the flatten function either way

    nij = nij[:S,:S]
    nyi = nyi[:T,:S]
    t_hat = get_estimator(nij)
    e_hat = get_estimator(nyi)
    q_hat = get_q_hat(x, xv)
    del xv[PADDING_CONST]
    del yv[PADDING_CONST]
    return t_hat, e_hat, q_hat


def get_q_hat(X, xvals):
    prefix_X = np.asarray([seq[0] for seq in X])
    q = np.zeros(len(xvals))
    for x in prefix_X:
        q[xvals[x]] += 1
    q = q[:len(xvals)-1]
    return q / q.sum()


def flatten_list(lst, PADDING_CONST=''):
    return np.asarray([item for sublist in lst for item in [PADDING_CONST] + sublist + [PADDING_CONST]])


def get_estimator(n_mat):
    denom = repmat(np.sum(n_mat, axis=0)[np.newaxis, :], n_mat.shape[0], 1)
    mask = denom == 0
    denom[mask] = 1
    res = np.divide(n_mat, denom)
    res[mask] = 0
    return res


def sample(Ns, xvals, yvals, t, e, q):
    """
    sample sequences from the model .
    : param Ns : a vector with desired sample lengths , a sample is generated per
    entry in the vector , with corresponding length .
    : param xvals : the possible values for x variables , ordered as in t , and e ( Dict - see MLE)
    : param yvals : the possible values for y variables , ordered as in e ( Dict - see MLE)
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : x , y - two iterables describing the sampled sequences .
    """
    seq_x = [''] * len(Ns)
    seq_y = [''] * len(Ns)
    for i, seq_len in enumerate(Ns):
        seq_x[i], seq_y[i] = sample_seq(seq_len, xvals, yvals, t, e, q)
    return seq_x, seq_y


def sample_seq(seq_len, xvlist, yvlist, t, e, q):
    # x1=sample type of first word from q
    # sample y1=yvals(sample a word from e(:, x1)
    # for rest of sentence:
    #      sample xi from t(x(i-1),:)
    #      sample yi like y1

    def sample_y(i):
        # if np.sum(e[:, i]) != 1.0:
        #     print (np.sum(e[:, i]))
        return choice(a=yvlist, size=1, p=e[:, i])[0]

    def sample_newx(i):
        # if np.sum(t[:, i]) != 1.0:
        #     print (np.sum(t[:, i]))
        return np.where(multinomial(1, t[:, i]) == 1)[0][0]

    seq_x = [''] * seq_len
    seq_y = [''] * seq_len
    xi = np.where(multinomial(1, pvals=q) == 1)[0][0]
    seq_x[0] = xvlist[xi]
    seq_y[0] = sample_y(xi)

    for i in range(1, seq_len):
        xi = sample_newx(xi)
        seq_x[i] = xvlist[xi]
        seq_y[i] = sample_y(xi)

    return seq_x, seq_y


def viterbi_non_general(y, suppxList, suppyDict, t, e, q):
    """
    Calculate the maximum a - posteriori assignment of x ’s .
    : param y : a sequence of words
    : param suppx : the support of x ( what values it can attain )
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : xhat , the most likely sequence of hidden states ( parts of speech ).
    """
    M, S = len(y), len(suppxList)
    v_mat = np.zeros((M, S, 2))
    v_mat[0, :, 1] = np.multiply(q, e[suppyDict[y[0]], :])

    for tidx in range(1, M):
        # calc row tidx
        for j in range(S):
            possible_values = np.asarray([v_mat[tidx - 1, i, 1] * t[i, j] * e[suppyDict[y[tidx]], j] for i in range(S)])
            v_mat[tidx, j, 0] = np.argmax(possible_values)
            v_mat[tidx, j, 1] = possible_values[int(v_mat[tidx, j, 0])]
    max_v_idx_cur = int(np.argmax(v_mat[-1, :, 1]))
    x_hat = np.zeros((M,1))
    for i in range(M):
        x_hat[-i-1] = max_v_idx_cur
        max_v_idx_cur = int(v_mat[M-1-i, max_v_idx_cur, 0])
    return np.asarray(suppxList)[x_hat.astype(np.int32)][:,0]


def viterbi(y, suppxList, phi, w):
    """
    Calculate the assignment of x that obtains the maximum log - linear score .
    : param y : a sequence of words
    : param suppx : the support of x ( what values it can attain )
    : param phi : a mapping from ( x_t , x_ { t +1} , y_ {1.. t +1} to indices of w
    : param w : the linear model
    : return : xhat , the most likely sequence of hidden states ( parts of speech ).
    """

    def normalize_row(r): return r / r.sum()

    M, S = len(y), len(suppxList)

    v_mat = np.zeros((M, S, 2))
    v_mat[0, :, 1] = normalize_row(np.exp(np.asarray([np.sum(w[phi(xt, 'XXX', y, 0)]) for xt in suppxList])))

    from time import time
    t = time()
    phi_trans = np.zeros((S, S))  # Every row is a specific x(t-1), every column is a choice for x(t)
    for tidx in range(1, M):
        # calc row tidx
        phi_trans[:] = 0  # Every row is a specific x(t-1), every column is a choice for x(t)

        for idx, xt in enumerate(suppxList):
            phi_trans[:, idx] = np.exp(np.asarray([np.sum(w[phi(xt, xprev, y, tidx)]) for xprev in suppxList]))

        phi_trans = phi_trans * (v_mat[tidx - 1, :, 1] / phi_trans.sum(axis=1))[:, np.newaxis]

        v_mat[tidx, :, 0] = phi_trans.argmax(axis=0)
        v_mat[tidx, :, 1] = phi_trans.max(axis=0)
    print(str(M), time()-t)
    max_v_idx_cur = int(np.argmax(v_mat[-1, :, 1]))
    x_hat = np.zeros((M, 1))
    for i in range(M):
        x_hat[-i - 1] = max_v_idx_cur
        max_v_idx_cur = int(v_mat[M - 1 - i, max_v_idx_cur, 0])
    return np.asarray(suppxList)[x_hat.astype(np.int32)][:, 0]


def perceptron(X, Y, suppxList, phi, w0, rate):
    """
    Find w that maximizes the log - linear score
    : param X : POS tags for sentences ( iterable of lists of elements in suppx )
    : param Y : words in respective sentences ( iterable of lists of words in suppy )
    : param suppx : the support of x ( what values it can attain )
    : param suppy : the support of y ( what values it can attain )
    : param phi : a mapping from ( None | x_1 , x_2 , y_2 to indices of w
    : param w0 : initial model
    : param rate : rate of learning
    : return : w , a weight vector for the log - linear model features .
    """
    def update_w(x_hat, i, w):
        for t in range(len(X[i])):
            w[phi(X[i][t], X[i][t-1], Y[i], t)] += rate
            w[phi(x_hat[t], x_hat[t-1], Y[i], t)] -= rate

    N = len(X)
    w = w0.copy()
    for i in range(300):
        x_hat = viterbi(Y[i], suppxList, phi, w)
        update_w(x_hat, i, w)

    return w
