# README for sig_test_procedures
## Description:
There are `8` Python files in total:
* `fileReader.py`: read the input score file
* `testCase.py`: the obeject class for testCase
* `dataAnalysis.py`: conduct pre-testing exploratory data analysis, split the score into eval units and check for test assumptions
* `sigTesting.py`: run significance testing
* `effectSize.py`: choose an effect size estimator/index and estimate effect size
* `powerAnalysis.py`: conduct post-testing power analysis

This is one config file `config.yml`. The parameters to specify are:
* eval_unit_size
* eval_unit_met
* shuffled
* random_seed
* normal_test_alpha

* sig_test_alpha
* sig_boot_perm_B

* eff_size_ind

* power_alpha
* power_method
* power_dist_name
* power_num_of_sim
* power_B

* fig_output_dir
* report_output_dir

The following parameters are system specifications:
* MIN_eval_unit_size
* MIN_normal_test_alpha
* MAX_normal_test_alpha

* MIN_sig_test_alpha
* MAX_sig_test_alpha

* MAX_sig_boot_perm_B

* MIN_power_alpha
* MAX_power_alpha

* MIN_power_samplesize
* MIN_power_num_of_sim
* MAX_power_B

* mu

There is also a `main.py` script to test functions of the above 8 scripts, where the arguments are:
* `<sys.argv[1]>` = `score_file`
* `<sys.argv[2]>` = `config.yml`

## Input file:
The input file should be two columns of scores, separated by whitespace, like the following:

> 0.1352 0.1552

> 0.5326 0.2356

> 0.2672 0.2534

> ....

I included a test score file called *score*, which has 2001 rows of BLEU scores for 2 MT systems from WMT 2018. 

### Reading input file
The script `fileReader.py` will read the input score file line by line, split each line by whitespace and save the two scores into two dictionaries. The two scores are named `score1` and `score2`.

## testCase class:
The class `testCase` is an object that corresponds to a one-time testing run. It has the following attributes:
* `score1`: *dict*
* `score2`: *dict*
* `score_diff`: dict
* `testParam`: *string*, parameter to test (mean or median)
* `sigTest`: *string*, the name of the significance test to run
* `effSize`: *(float,string)*, effect size estimate and estimator name
* `powAnaly`: *dict*, post-power against different sample sizes

There is a built-in function:
* `calc_score_diff()`: calculates score difference


## Stage 2: data analysis
The script `dataAnalysis.py` conducts the pre-testing exploratory data analysis to plot histograms and check for test assumptions.

### Partitioning score difference:
The function `partition_score(score1, score2, score_diff, eval_unit_size, shuffled, randomSeed, method, output_dir)` splits *score1, score2, score_diff* into evaluation units, of which the size is specified by the user. The user can also specify whether they want to reshuffle first, the seed used for reshuffling and the method they want to use for calculation (mean or median). This function will also plot the histogram of the partitioned *score1*, *score2* and *score_diff*. Note that after partitioning, the original *score1* and *score2* are overwritten with the new scores.


### Skewness check:
The function `skew_test(score)` checks whether the distribution of *score_diff* is skewed in order to determine a good measure for central tendency. Note that here mean or median has nothing to do with the method of calculation in evaluation unit partitioning. The rules of thumb are:
1. abs(skewness) > 1: highly skewed, use `median`.
2. 0.5 <= abs(skewness) < 1: moderately skewed, use `median`.
3. abs(skewness) < 0.5: roughly symmetric, use `mean` or `median`.
If skewed, then the distribution is not normal for sure.

### Normality test:
The function `normality_test(score,alpha)` will conduct Shapiro-Wilks normality test for *score_diff* at a specified significance level `alpha`. The return value is a boolean, where `True` indicates normality and `False` indicates non-normality. 


### Recommending significance tests:
The function `recommend_test` recommends a list of significance tests based on the results given before (from functions `skew_test` and `normality_test`):
1. If normal, use `t test` (other tests are also applicable but have lower power) <`t`>
2. If not normal but `mean` is a good measdure for central tendency, use:
    1. bootstrap test bassed on mean (t ratios) or medians <`bootstrap`>
    2. sign test <`sign`>
    3. sign test calibrated by permutation (based on mean or median) <`permutation`>
    4. Wilcoxon signed rank test <`wilcoxon`>
    5. t test (may be okay for large samples) <`t`>
3. If not normal and highly skewed, use:
    1. bootstrap test based on median <`bootstrap_med`>
    2. sign test <`sign`>
    3. sign test calibrated by permutation (based on median) <`permutation_med`>
    4. Wilcoxon signed rank test <`wilcoxon`>

## Stage 3: significance testing
The script `sigTesting.py` contains functions to run the significance testing chosen in Stage 2.


## Stage 4: reporting (effect size estimation and power analysis)
The scripts `effectSize.py` and `powerAnalysis.py` partially provide functionalities for Stage 4. The user needs to specify what effect size index to use and what power analysis method (*bootstrap* or *montecarlo*) to use (currently only normal distribution is implemented for the Monte Carlo method).

## `main.py` test case example:
In this script, I choose the significance test to be the second in the list. If `eval_unit_size` is different (say 5), then the list of test might have different length, which may give rise to some bugs. 

Note that the power analysis part may take relatively longer time to complete.

For example, run the following:

`python main.py score config.yml`
