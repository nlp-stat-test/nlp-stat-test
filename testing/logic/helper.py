def helper(function_name: str) -> str:
    advice =\
    {
        "read_score_file": "The input file must contain two columns of numerical data points of the same length.",
        "plot_hist": "This function plots the histograms of the two input samples individually.",
        "plot_hist_diff" : "This function plots the histogram of the pairwise differences between the two input samples.",
        "partition_score" : "This function divides the pairwise differences into evaluation units of which the size is specified by the user. Leftover data points are discarded.",
        "normality_test" : "This function conducts the Shapiro-Wilks normality test for an input score. The default alpha level is 0.05.",
        "skew_test" : "This function estimates the sample skewness of the input score, based on which it recommends the measure for central tendency (mean or median).",
        "recommend_test" : "This function, based on the results of the normality test and skewness check, recommends a list of signifiance tests that are appropriate.",
        "calc_eff_size" : "This function calculates the effect size estimate based on the significance test used and the testing parameter recommended.",
        "cohend" : "This function calculates the Cohen's d effect size estimator.",
        "hedgesg" : "This function takes the Cohen's d estimate as an input and calculates the Hedges's g.",
        "wilcoxon_r" : "This function calculates the standardized z-score (r) for the Wilcoxon signed-rank test.",
        "hodgeslehmann" : "This function estimates the Hodges-Lehmann estimator for the input score.",
        "run_sig_test" : "The function runs the statistical significance test recommended by the data analysis and returns the value of the statistic, the p-value and the rejection conclusion. For bootstrap tests, this function only returns a confidence interval and a rejection conclusion.",
        "bootstrap_test" : "This function conducts the bootstrap test based either on the t ratio or on the median.",
        "permutation_test" : "This function conducts the sign test calibrated by permutation based either on the mean difference or the median difference.",
        "post_power_analysis" : "This function conducts retrospective power analysis. If the distribution is known (normal), it uses Monte Carlo simulation; if not, then it uses bootstrap method.",
    }
    return advice[function_name]
