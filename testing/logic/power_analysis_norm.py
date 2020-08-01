"""
Power analysis for normal data

Description: this script is for a simple illustration of power analysis limited to
data that are known a priori normally distributed with known mean and known sd. 
We conduct power analysis using Monte Carlo simulation which directly simulates 
data points from the known distribution for many times (B) with different sample size, 
do one-sample t test and count how many times the tests reject the null. 
The final power curve w.t.r sample size is thus obtained.
"""

# imports
import os
import sys
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def prosp_power_analysis_norm(mu, sd, pow_lev=0.8, alpha=0.05, B=200, step_size=5):
    """
    This is a function fot prospective power analysis for normal data
    based on Monte Carlo simulation. We only consider doing one sample testing
    in this case, and the null hypothesis is that the population mean is 0.

    @param mu: population mean
    @param sd: population standard deviation
    @param pow_lev: desired power level (default 0.8)
    @param alpha: significance level or type II error (default 0.05)
    @param B: number of iterations for each sample size (default 200)
    @param step_size: step size for each sample size (default 5)

    @return: a dictionary (keys: different sample sizes, values: simulated power)
    @return: the minimal sample size to reach the power level
    """
    power = {} # initialize dict for power
    power_temp = 0 # temporary power level
    samp_size = 5 # starting sample size (t test will run into errors for small sample size)
    while power_temp < pow_lev: # while the desired power is not reached
        pval=[] # initialize p value vector
        for b in range(0,B-1):
            x_b = np.random.normal(loc=mu,scale=sd,size=samp_size) # sample from normal dist
            pval.append(stats.ttest_1samp(x_b,0).pvalue) # one sample t test for mu=0
        power_temp = float(len([j for j in pval if j < alpha]))/B # calculating power
        power[samp_size] = power_temp
        pval=[]
        samp_size+=step_size
    return([power,list(power)[-1]]) 

def plot_power_curve(mu, sd, pow_lev=0.8):
    """
    This is a function to plot the simnulated power curve by calling the power analysis
    function with only mu and sd specified.

    @param mu: population mean
    @param sd: population standard deviation
    @param pow_lev: desired power level (default 0.8)
    @return: a figure
    """
    sys.stderr.write("##### Starting Monte Carlo simulation for power #####\n")
    power_dict = prosp_power_analysis_norm(mu, sd)[0]
    sys.stderr.write("----- Finished power analysis -----\n")
    sys.stderr.write("##### Saving figures #####\n")
    fig = plt.figure()
    plt.plot(np.array(list(power_dict.keys())),np.array(list(power_dict.values())))
    plt.plot(np.array(list(power_dict.keys())),[pow_lev]*len(np.array(list(power_dict.keys()))),'r--')

    fig.suptitle('Power Curve for Different Sample Sizes', fontsize=20)
    plt.xlabel('Sample Size', fontsize=18)
    plt.ylabel('Power', fontsize=16)
    plt.savefig('power_sample_size.png')
    sys.stderr.write("----- Finished saving figures -----\n")


if __name__ == '__main__':
    mu= float(sys.argv[1])
    sd = float(sys.argv[2])
    #alpha = str(sys.argv[4])
    #B = bool(sys.argv[5])
    #step_size = int(sys.argv[6])

    plot_power_curve(mu, sd)
    


