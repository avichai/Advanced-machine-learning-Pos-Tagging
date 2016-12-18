import numpy as np
from numpy.matlib import repmat as repmat
from numpy.random import multinomial, choice

EPS = 1E-50


def log_likelihood(q, t, e, ni, nij, nyi):
    """
    gets the log likelihood for HMM
    :param q: shape = | val ( X )| probability to see i in valX as the first word in
                the sentence.
    :param t: shape = (| val ( X )| ,| val ( X )|) transition
    :param e: shape = (| val ( X )| ,| val ( Y )|) emission
    :param ni: number of times i in val(X) began a sequence - sum(ni)=#sentences
    :param nij: number of times i in val(X) appeared before j in val(X)
    :param nyi: number of times y in val(Y) was omitted in state i in val(X)
    :return: the log likelihood for HMM
    """
    # The addition of EPS will make almost no difference where q(i)!=0. Where q(i)==0, it means n(i)==0 so
    # either way it will make no difference after the dot product
    return (np.dot(ni, np.log(q + EPS)) + np.dot(nij.flatten(), np.log(t + EPS).flatten()) +
            np.dot(nyi.flatten(), np.log(e + EPS).flatten())) / np.sum(ni)


def get_ni(X, xvalsDict):
    """
    gets the number of times i in xvals began a sequence
    :param X: an iterable over sequences of POS tags
    :param xvalsDict: a dictionary of POS tags
    :return:
    """
    q = np.zeros(len(xvalsDict))
    for seq in X:
        q[xvalsDict[seq[0]]] += 1
    return q


def get_ni_nij_nyi(X, y, xvDict, yvDict):
    """
    gets the number of times i in xv began a sequence,
    the number of times i in xv appeared before j in xv,
    number of times y in yv was omitted in state i in xv
    :param X: an iterable over sequences of POS tags
    :param y: a matching iterable over sequences of words
    :param xvDict: A dictionary from POS tagging (possible xvalue) to index between 0 to len(xv)-1 - Note that its a
                bijective mapping
    :param yvDict: A dictionary from Possible word (possible xvalue) to index between 0 to len(yv)-1 - Note that its a
                bijective mapping
    :return: the number of times i in xv began a sequence,
                the number of times i in xv appeared before j in xv,
                number of times y in yv was omitted in state i in xv
    """
    PADDING_CONST = '##$$##'  # A padding constant which is not in either supports of x or y
    while PADDING_CONST in yvDict or PADDING_CONST in xvDict:
        PADDING_CONST += '$'

    flatten_X, flatten_Y = flatten_list(X, PADDING_CONST), flatten_list(y, PADDING_CONST)
    M = len(flatten_Y)
    S, T = len(xvDict), len(yvDict)
    nij = np.zeros((S + 1, S + 1))
    nyi = np.zeros((T + 1, S + 1))
    xvDict[PADDING_CONST] = S
    yvDict[PADDING_CONST] = T
    for i in range(M - 1):
        nij[xvDict[flatten_X[i]], xvDict[flatten_X[i + 1]]] += 1
        nyi[yvDict[flatten_Y[i]], xvDict[flatten_X[i]]] += 1
        # Note that we don't go over the last pair of y and x, but its a padding from the flatten function either way

    nij = nij[:S, :S]
    nyi = nyi[:T, :S]
    del xvDict[PADDING_CONST]
    del yvDict[PADDING_CONST]
    ni = get_ni(X, xvDict)

    return ni, nij, nyi


def mle(x, y, xvDict, yvDict):
    """
    Calculate the maximum likelihood estimators for the transition and
    emission distributions , in the multinomial HMM case .
    :param x: an iterable over sequences of POS tags
    :param y: a matching iterable over sequences of words
    :param xvDict: A dictionary from POS tagging (possible xvalue) to index between 0 to len(xv)-1 - Note that its a
                bijective mapping
    :param yvDict: A dictionary from Possible word (possible xvalue) to index between 0 to len(yv)-1 - Note that its a
                bijective mapping
    :return: a  tuple (t, e, q), with
            t . shape = (| val ( X )| ,| val ( X )|) , and
            e . shape = (| val ( X )| ,| val ( Y )|)
            q . TODO
    """
    ni, nij, nyi = get_ni_nij_nyi(x, y, xvDict, yvDict)
    t_hat = get_estimator(nij)
    e_hat = get_estimator(nyi)
    q_hat = ni / ni.sum()
    return t_hat, e_hat, q_hat, log_likelihood(q_hat, t_hat, e_hat, ni, nij, nyi)


def flatten_list(lst, PADDING_CONST=''):
    """
    flatten the list of sentences using a padding between them.
    :param lst: the list to flatten
    :param PADDING_CONST: the padding between the list items
    :return: flatten the list of sentences using a padding between them.
    """
    return np.asarray([item for sublist in lst for item in [PADDING_CONST] + sublist + [PADDING_CONST]])


def get_estimator(n_mat):
    """
    get the estimator of t_hat or q_hat while ignoring 0's
    :param n_mat: the from which we get the estimator from.
    :return: the estimator of t_hat or q_hat while ignoring 0's
    """
    denom = repmat(np.sum(n_mat, axis=0)[np.newaxis, :], n_mat.shape[0], 1)
    mask = denom == 0.
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
    """
    sample one sentence
    :param seq_len: the length of the sentence we wish to sample
    :param xvlist: the possible values for x variables , ordered as in t , and e ( Dict - see MLE)
    :param yvlist: the possible values for y variables , ordered as in e ( Dict - see MLE)
    :param t: shape = (| val ( X )| ,| val ( X )|) transition
    :param e: shape = (| val ( X )| ,| val ( Y )|) emission
    :param q: shape = | val ( X )| probability to see i in valX as the first word in
                the sentence.
    :return:
    """
    def sample_y(col):
        """
        sample one word
        :param col:  index of the current POS
        :return: sampled one word
        """
        return choice(a=yvlist, size=1, p=e[:, col])[0]

    def sample_newx(col):
        """
        sample a tag
        :param col: index of the current POS
        :return: sample a tag
        """
        return np.where(multinomial(1, t[:, col]) == 1)[0][0]

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
    : param q : shape = | val ( X )| probability to see i in valX as the first word in
                the sentence.
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
    x_hat = np.zeros((M, 1))
    for i in range(M):
        x_hat[-i - 1] = max_v_idx_cur
        max_v_idx_cur = int(v_mat[M - 1 - i, max_v_idx_cur, 0])
    return np.asarray(suppxList)[x_hat.astype(np.int32)][:, 0]


def viterbi(y, suppxList, phi, w):
    """
    Calculate the assignment of x that obtains the maximum log - linear score .
    : param y : a sequence of words
    : param suppx : the support of x ( what values it can attain )
    : param phi : a mapping from ( x_t , x_ { t +1} , y_ {1.. t +1} to indices of w
    : param w : the linear model
    : return : xhat , the most likely sequence of hidden states ( parts of speech ).
    """

    def normalize_row(r):
        """
        normalize a row vector.
        :param r: row vector.
        :return: normalized a row vector.
        """
        return r / r.sum()

    M, S = len(y), len(suppxList)

    v_mat = np.zeros((M, S, 2))
    v_mat[0, :, 1] = normalize_row(np.exp(np.asarray([np.sum(w[phi(xt, 'XXX', y, 0)]) for xt in suppxList])))

    from time import time
    t = time()
    phi_trans = np.zeros((S, S))  # Every row is a specific x(t-1), every column is a choice for x(t)
    for tidx in range(1, M):
        # calc row tidx
        phi_trans[:] = 0  # Every row is a specific x(t-1), every column is a choice for x(t)

        for col, xt in enumerate(suppxList):
            for row, xprev in enumerate(suppxList):
                phi_trans[row, col] = np.sum(w[phi(xt, xprev, y, tidx)])
        phi_trans = np.exp(phi_trans)
        phi_trans = phi_trans * (v_mat[tidx - 1, :, 1] / phi_trans.sum(axis=1))[:, np.newaxis]

        v_mat[tidx, :, 0] = phi_trans.argmax(axis=0)
        v_mat[tidx, :, 1] = phi_trans.max(axis=0)
    #print(str(M), time() - t)
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
        """
        update the weights vector
        :param x_hat: the estimator for POS tags
        :param i: the index in the sentence
        """
        for t in range(len(X[i])):
            w[phi(X[i][t], X[i][t - 1], Y[i], t)] += rate
            w[phi(x_hat[t], x_hat[t - 1], Y[i], t)] -= rate

    N = len(X)
    w = w0.copy()
    for i in range(10):
        x_hat = viterbi(Y[i], suppxList, phi, w)
        update_w(x_hat, i, w)

    return w
