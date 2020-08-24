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
import numpy as np
from scipy import stats
import sys




def prosp_power_analysis_norm(mu_diff, sd, pow_lev, alpha, alternative):
    """
    This is a function fot prospective power analysis for normal data
    based on Monte Carlo simulation. We only consider doing one sample testing
    in this case, and the null hypothesis is that the population mean is 0.

    @param mu: mean difference
    @param sd: population standard deviation
    @param pow_lev: desired power level
    @param alpha: significance level or type II error (default 0.05)

    @return: required minimal sample size
    """



    delta = (sd**2)/(mu_diff**2)


    if alternative == "less" or "greater":
        n = ((stats.norm.ppf(1-alpha)+stats.norm.ppf(pow_lev))**2)*delta

    if alternative == "two-sided":
        n = ((stats.norm.ppf(1-alpha/2)+stats.norm.ppf(pow_lev))**2)*delta


    return(np.ceil(n))



if __name__ == '__main__':
    mu_diff = float(sys.argv[1])
    sd = float(sys.argv[2])
    pow_lev = float(sys.argv[3])     
    alpha = float(sys.argv[4])
    h1= str(sys.argv[5])

    n = prosp_power_analysis_norm(mu_diff, sd, pow_lev, alpha, h1)

    print(n)


    
    


