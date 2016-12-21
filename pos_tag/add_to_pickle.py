import sys
from pos_tag.test_pos_tagging import TestingResult
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    with open('bk/results_10_001.pickle', 'rb') as fid: res10_001 = pickle.load(fid, encoding='Latin1').perceptron_curves['0.1']
    with open('bk/results_10_005.pickle', 'rb') as fid: res10_005 = pickle.load(fid, encoding='Latin1').perceptron_curves['0.1']
    with open('bk/results_20_001.pickle', 'rb') as fid: res20_001 = pickle.load(fid, encoding='Latin1').perceptron_curves['0.2']

    # res30__03
    intervals = [    0  ,1000  ,2000 , 3000  ,4000  ,5000 , 6000  ,7000 , 8000 , 9000 ,10000 ,11000,
 12000 ,13000 ,14000 ,14600]
    res30_003 = pd.DataFrame({'simple': pd.Series(data=[ 0.99038462 , 0.26211782  ,0.21990309 , 0.16235378 , 0.18859971 , 0.16240945,
  0.14208341 , 0.16539162 , 0.12364348  ,0.11352657 , 0.11488203 , 0.12240741,
  0.1340507 ,  0.12315095 , 0.10373832 , 0.10300587], index=intervals),
                                'char': pd.Series(data=[ 0.97165617 , 0.76523903 , 0.77824347 , 0.7443038  , 0.83311555  ,0.79332182,
  0.7659653 ,  0.79687219 , 0.79230916 , 0.80714418  ,0.72705403 , 0.73829149,
  0.78536934  ,0.71304025 , 0.75803403 , 0.75804039], index=intervals),
                                'complex': pd.Series(data=[ 1.    ,      0.22467192 , 0.21819111 , 0.16562778 , 0.15837601 , 0.15833637,
  0.15233892 , 0.13905895 , 0.1434629 ,  0.17326733 , 0.11883692 , 0.13004086,
  0.13055454 , 0.14480112 , 0.11252236 , 0.11675217], index=intervals)})

    # res30__05 AVG
    intervals2 = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000,
                 12000, 13000, 14000, 14589]
    res30_005 = pd.DataFrame(
        {'AvergedWeightComplex': pd.Series(data=[ 0.99619496 , 0.19476579,  0.13857251 , 0.12944523  ,0.11804788  ,0.10948638,
  0.10342941 , 0.095898  ,  0.0970297 ,  0.08748143 , 0.08121278,  0.07983857,
  0.07696317  ,0.07805409  ,0.07024648  ,0.06741113], index=intervals2)})


    plt.figure()
    ax = plt.subplot(2,2,1)
    res10_001.plot(legend=True, title='Train Size: {0}%, Rate: {1}, Return Type: Last-W'.format(10, 0.01), marker='o', yticks=np.arange(0.,1.,0.05), ax=ax)
    ax = plt.subplot(2,2,3)
    res10_005.plot(legend=True, title='Train Size: {0}%, Rate: {1}, Return Type: Last-W'.format(10, 0.05), marker='o', yticks=np.arange(0.,1.,0.05), ax=ax)
    ax = plt.subplot(2,2,4)
    res30_003.plot(legend=True, title='Train Size: {0}%, Rate: {1}, Return Type: Last-W'.format(30, 0.03), marker='o', yticks=np.arange(0.,1.,0.05), ax=ax)
    ax = plt.subplot(2,2,2)
    res20_001.plot(legend=True, title='Train Size: {0}%, Rate: {1}, Return Type: Last-W'.format(20, 0.01), marker='o', yticks=np.arange(0.,1.,0.05), ax=ax)


    res30_005.plot(legend=True, title='Train Size: {0}%, Rate: {1}, Return Type: Averaged-W'.format(30, 0.05), marker='o', yticks=np.arange(0.,1.,0.05))

    plt.show(block=True)

if __name__ == '__main__':
    main()
