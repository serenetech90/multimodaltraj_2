import matplotlib
import numpy as np
from scipy.stats import *
import scipy as sc
import math

def main():
    f = '/home/serene/PycharmProjects/multimodaltraj/kernel_models/MX-LSTM-master/data/annotation_tc.txt'
             # 'r')

    mat = np.genfromtxt(f, delimiter=',')
    headPose = mat[:,12]
    loc_x = np.power(np.subtract(mat[:,8], mat[:,10]), 2.0)
    loc_y = np.power(np.subtract(mat[:,9], mat[:,11]), 2.0)

    # MX-LSTM assumes gaussian distribution of trajectories and that is why they use pearson correlation test
    corr, _ = pearsonr(headPose, loc_y)
    print('Linear correlation = ' , corr)

    # For non-Gaussian assumptions
    corr, _ = spearmanr(headPose, loc_y)
    print('Non-Linear correlation = ', corr)
    return

    # Negatively correlated features in both linear and non-linear tests
    #
if __name__ == '__main__':
    main()