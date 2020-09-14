# imports
import numpy as np
from scipy import stats


def prosp_power_analysis_norm(d, sigma, pow_lev, alpha, direction):
    """
    This function conducts pre-testing power analysis and
    calculates the minimally required sample size for a normal sample.

    @param d: difference between the mean differences under H1 and H0
    @param sigma: standard deviation
    @param pow_lev: power level
    @param alpha: significance level
    @param direction: direction of the test, two-sided or one-sided
    @return: required minimal sample size
    """
    # first calculates for a z test

    n_z = np.ceil(z_test_sample_size(d, sigma, alpha, pow_lev, direction))

    # first iteration for t test

    n_t_1 = np.ceil(t_test_sample_size(d, sigma, n_z-1, alpha, pow_lev, direction))


    # second iteration for t test

    n_t_2 = np.ceil(t_test_sample_size(d, sigma, n_t_1-1, alpha, pow_lev, direction))

    return(np.ceil(n_t_2 ))



def z_test_sample_size(d, sigma, alpha, pow_lev, direction):
    if direction == "one-sided":
        n = ((stats.norm.ppf(1-alpha)+stats.norm.ppf(pow_lev))**2)*(float(sigma)/float(d))**2
    elif direction == "two-sided":
        n = ((stats.norm.ppf(1-alpha/2)+stats.norm.ppf(pow_lev))**2)*(float(sigma)/float(d))**2
    return(n) 


def t_test_sample_size(d, sigma, dof, alpha, pow_lev, direction):
    if direction == "one-sided":
        n = ((stats.t.ppf(1-alpha, df=dof)+stats.t.ppf(pow_lev, df=dof))**2)*(float(sigma)/float(d))**2
    elif direction == "two-sided":
        n = ((stats.t.ppf(1-alpha/2, df=dof)+stats.t.ppf(pow_lev, df=dof))**2)*(float(sigma)/float(d))**2
    return(n) 




