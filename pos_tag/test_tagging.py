from pos_tag import pos_tagging
from pos_tag import parser


def test_mle():
    data, xv, yv = parser.collect_sets('data_split.gz', k=10, n=1) #9/10 of data is for training, only one copy
    for perc in [0.1, 0.25, 0.5, 0.9]:
        t_hat, e_hat = pos_tagging.mle(xv[:int(len(xv) * perc)], yv[:int(len(yv) * perc)])



def main():
    try:
        for test in [test_mle]:
            test()
    except Exception as e:
        print("Failed test due to : {0}".format(e))
        exit(-1)

if __name__ == '__main__':
    main()