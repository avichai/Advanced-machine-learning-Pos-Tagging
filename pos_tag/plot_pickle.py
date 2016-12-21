import sys
from pos_tag.test_pos_tagging import TestingResult
import pickle



def main():
    if len(sys.argv) != 2:
        raise Exception("Bad usage: python plot_pickle <pickle_path>")
    try:
        with open(sys.argv[1], 'rb') as fid: res = pickle.load(fid, encoding='Latin1')
    except Exception:
        with open(sys.argv[1], 'rb') as fid:
            res = pickle.load(fid)
    res.plot()

if __name__ == '__main__':
    main()
