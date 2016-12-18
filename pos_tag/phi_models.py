import numpy as np


def get_hmm_w_vec(t, e, q, suppX, suppY):
    """

    :param t:
    :param e:
    :param q:
    :param suppX:
    :param suppY:
    :return:
    """
    S, T = len(suppX), len(suppY)
    D = S + S * S + T * S
    w = np.zeros(D)
    EPS = 1E-50
    w[:S] = np.log(q + EPS)
    w[S:S + S * S] = np.log(t.flatten() + EPS)
    w[S + S * S:] = np.log(e.flatten() + EPS)
    return w

def get_hmm_phi(suppX, suppY):
    S, T = len(suppX), len(suppY)
    D = S + S * S + T * S
    def simple_phi(xt, xprev, y, t):
        """
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
        """
        indices_vec = [0] * 2
        xt_idx = suppX[xt]
        yidx = suppY[y[t]]
        if t == 0:
            indices_vec[0] = xt_idx
            # feature_vec[xt_idx] = 1
        else:
            xprev_idx = suppX[xprev]
            indices_vec[0] = S + xprev_idx * S + xt_idx
        indices_vec[1] = S + S * S + yidx * S + xt_idx
        return indices_vec
    return simple_phi, D


def get_word_carachteristics_phi():
    """

    :return:
    """
    num_features = 5

    def phi(xt, xprev, y, t):
        indices_vec = np.array([y[t][0].isupper(),
                                np.count_nonzero([c.isupper() for c in y[t]]) > 1,
                                y[t].endswith('ed'),
                                y[t].endswith('ing'),
                                all(c.isalpha() for c in y[t])], dtype=np.bool)
        return np.where(indices_vec == True)[0].tolist()
    return phi, num_features


def get_complex_phi(phi1, D1, phi2, D2):
    """

    :param phi1:
    :param D1:
    :param phi2:
    :param D2:
    :return:
    """
    def phi(xt, xprev, y, t):
        return phi1(xt, xprev, y, t) + (np.asarray(phi2(xt, xprev, y, t)) + D1).tolist()
    return phi, D1+D2
