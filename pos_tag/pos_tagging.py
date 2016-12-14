import numpy as np
from numpy.matlib import repmat as repmat
from numpy.random import multinomial, choice

MAX_OCC_MAT_SIZE = 0.25e9  # Maximum allowed matrix elements


def mle(x, y, xv, yv):
    '''
        Calculate the maximum likelihood estimators for the transition and
        emission distributions , in the multinomial HMM case .
        : param x : an iterable over sequences of POS tags
        : param : a matching iterable over sequences of words
        : return : a     tuple (t , e ) , with
        t . shape = (| val ( X )| ,| val ( X )|) , and
        e . shape = (| val ( X )| ,| val ( Y )|)
    '''
    flatten_X, flatten_Y = flatten_list(x), flatten_list(y)
    M = len(flatten_Y)
    # TODO - assuming here memory can always handle matrix of (len(xv) X M)
    occ_mat_X = get_occ_mat(x, xv)
    occ_mat_Y = get_occ_mat(y, yv)
    nij = get_nij(occ_mat_X)
    nyi = get_nyi(occ_mat_X, occ_mat_Y)
    t_hat = get_estimator(nij)
    e_hat = get_estimator(nyi)
    return t_hat, e_hat, get_q_hat(x, xv)


def get_q_hat(X, xvals):
    prefix_X = np.asarray([seq[0] for seq in X])
    q = np.zeros(len(xvals))
    for i, j in enumerate(xvals):
        q[i] = np.sum((prefix_X == j).astype(np.uint32))
    return q / q.sum()


def flatten_list(lst):
    # TODO - make sure '' is not a possible x or y word!
    return np.asarray([item for sublist in lst for item in [''] + sublist + ['']])


def get_occ_mat(lst, vals):
    flat_list = flatten_list(lst)
    occ_mat = np.zeros((len(vals), len(flat_list)), dtype=np.bool)
    for i, v in enumerate(vals):
        occ_mat[i, :] = flat_list == v
    return occ_mat


def get_nij(occ_mat_X):
    return np.matmul(occ_mat_X[:, :-1].astype(np.uint32), np.transpose(occ_mat_X[:, 1:].astype(np.uint32)))


def get_nyi(occ_mat_X, occ_mat_Y):
    # TODO - make sure it is np.float64
    return np.matmul(occ_mat_Y.astype(np.uint32), occ_mat_X.transpose().astype(np.uint32))


def get_estimator(n_mat):
    # TODO - make sure it is np.float64
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
    : param xvals : the possible values for x variables , ordered as in t , and e
    : param yvals : the possible values for y variables , ordered as in e
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : x , y - two iterables describing the sampled sequences .
    """
    seq_x = [''] * len(Ns)
    seq_y = [''] * len(Ns)
    for i, seq_len in enumerate(Ns):
        seq_x[i], seq_y[i] = sample_seq(seq_len, xvals, yvals, t, e, q)
    return seq_x, seq_y


def sample_seq(seq_len, xvals, yvals, t, e, q):
    # x1=sample type of first word from q
    # sample y1=yvals(sample a word from e(:, x1)
    # for rest of sentence:
    #      sample xi from t(x(i-1),:)
    #      sample yi like y1

    xvlist = list(xvals)
    yvlist = list(yvals)

    def sample_y(i):
        # if np.sum(e[:, i]) != 1.0:
        #     print (np.sum(e[:, i]))
        return choice(a=yvlist, size=1, p=e[:, i])

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


def viterbi_non_general(y, suppx, suppy, t, e, q):
    """
    Calculate the maximum a - posteriori assignment of x ’s .
    : param y : a sequence of words
    : param suppx : the support of x ( what values it can attain )
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : xhat , the most likely sequence of hidden states ( parts of speech ).
    """
    M, S = len(y), len(suppx)
    yv = np.asarray(list(suppy))
    v_mat = np.zeros((M, S, 2))
    v_mat[0, :, 1] = np.multiply(q, e[np.where(yv == y[0])[0][0], :])

    for tidx in range(1, M):
        # calc row tidx
        for j in range(S):
            possible_values = np.asarray([v_mat[tidx - 1, i, 1] * t[i, j] * e[np.where(yv == y[tidx])[0][0], j] for i in range(S)])
            v_mat[tidx, j, 0] = np.argmax(possible_values)
            v_mat[tidx, j, 1] = possible_values[int(v_mat[tidx, j, 0])]
    max_v_idx_cur = int(np.argmax(v_mat[-1, :, 1]))
    x_hat = np.zeros((M,1))
    for i in range(M):
        x_hat[-i-1] = max_v_idx_cur
        max_v_idx_cur = int(v_mat[M-1-i, max_v_idx_cur, 0])
    return np.asarray(list(suppx))[x_hat.astype(np.int32)][:,0]


def viterbi(y , suppx , suppy, phi, w):
    """
    Calculate the assignment of x that obtains the maximum log - linear score .
    : param y : a sequence of words
    : param suppx : the support of x ( what values it can attain )
    : param phi : a mapping from ( x_t , x_ { t +1} , y_ {1.. t +1} to indices of w
    : param w : the linear model
    : return : xhat , the most likely sequence of hidden states ( parts of speech ).
    """

    def normalize_row(r): return r / r.sum()

    M, S = len(y), len(suppx)
    yv = np.asarray(list(suppy))
    v_mat = np.zeros((M, S, 2))
    v_mat[0, :, 1] = normalize_row(np.exp(np.asarray([np.dot(w, phi(xt, 1, y, 0)) for xt in S])))

    for tidx in range(1, M):
        # calc row tidx
        phi_trans = np.zeros((S,S))  # Every row is a specific x(t-1), every column is a choice for x(t)

        for xt in range(S):
            phi_trans[:, xt] = np.exp(np.asarray([np.dot(w, phi(xt, xprev, y, tidx)) for xprev in range(S)]))

        phi_trans = phi_trans / phi_trans.sum(axis=1)[:, np.newaxis]
        phi_trans = phi_trans * v_mat[tidx - 1, :, 1][:, np.newaxis]

        v_mat[tidx, :, 0] = phi_trans.argmax(axis=0)
        # v_mat[tidx, :, 1] = phi_trans[v_mat[tidx, :, 0].astype(np.int32)]
        v_mat[tidx, :, 1] = phi_trans.max(axis=0)

    max_v_idx_cur = int(np.argmax(v_mat[-1, :, 1]))
    x_hat = np.zeros((M, 1))
    for i in range(M):
        x_hat[-i - 1] = max_v_idx_cur
        max_v_idx_cur = int(v_mat[M - 1 - i, max_v_idx_cur, 0])
    return np.asarray(list(suppx))[x_hat.astype(np.int32)][:, 0]


def perceptron(X, Y, suppx, suppy, phi, w0, rate):
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
    def calc_change_weight(x_hat, i): np.sum([phi(X[i][t], X[i][t-1], Y[i], t) - phi(x_hat[t], x_hat[t-1], Y[i], t) for t in len(X[i])])

    N = len(X)
    w = np.zeros(N+1)
    w[0] = w0
    for i in range(N):
        x_hat = viterbi(Y[i], suppx, suppy, phi, w[i])
        w[i+1] += rate * calc_change_weight(x_hat, i)

    return np.mean(w[1:])
