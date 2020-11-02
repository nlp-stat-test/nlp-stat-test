# List of sig tests:
1. Student t test
2. Bootstrap test based on mean (t ratios)
3. Bootstrap test based on median
4. Sign test (exact test using binomial dist.)
5. Sign test  calibrated by permutation (mean)
6. Sign test calibrated by permutation (median)
7. Wilcoxon signed rank test


# List of effect size estimators (difference family):
1. Cohen’s d
2. Hedges’ g
3. Wilcoxon r score (z/\sqrt{n})
4. HL estimator (Hodges-Lehmann, median estimation for symmetric dist.)


# Power analysis methods:
1. Monte Carlo simulation (this assumes the distribution is known—e.g. normal)
    - estimate sample mean and sd (mu, sigma)
    - starting sample size (cannot be too small—e.g. 5)
    - step size (this is associated with the running time)
    - desired power level (e.g. 0.8)
    - the sig test  (e.g. t test)
    - the sig level, alpha (e.g. 0.05)
    - the null hypothesis (usually H0: mu=0)

    * this will draw a plot of the statistical power against the sample size.

2. Bootstrap (this does not assume known distribution; just use the data)
    - starting sample size
    - ending sample size (the actual sample size we have)
    - step size
    - desired power level (may not be achieved)
    - the sig test 
    - the sig level, alpha
    - the null hypothesis


