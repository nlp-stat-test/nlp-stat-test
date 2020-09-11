

"""
This is a script for pre-testing exploratory data analysis. Main functionality inclues:
1. read the score file (assumes a file with two columns of numbers, each row separated by whitespace)
2. plot histograms of score1, score2 and their difference--score_diff (assumes paired sample)
3. partition the difference
"""

# imports
import os
import sys
import numpy as np
from scipy import stats
import random
import matplotlib
matplotlib.use('Svg')
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
	


def partition_score(score1, score2, score_diff, eval_unit_size, shuffled, randomSeed, method, output_dir):
	"""
	This function partitions the score difference with respect to the given
	evaluation unit size. Also, the user can choose to shuffle the score difference
	before partitioning.

	@param score1, score2: original scores
	@param score_diff: score difference, a dictionary
	@param eval_unit_size: evaluation unit size
	@param shuffled: a boolean value indicating whether reshuffling is done
	@param method: how to calculate score in an evaluation unit (mean or median)
	@return: score1_new, score2_new, score_diff_new, the partitioned scores, dictionary
	"""
	ind = list(score_diff.keys()) # the keys should be the same for three scores

	if shuffled:
		ind_shuffled = random.Random(randomSeed).shuffle(ind)
	ind_shuffled = np.array_split(ind,np.floor(len(ind)/eval_unit_size))
	ind_new = 0

	score1_new = {}
	score2_new = {}
	score_diff_new = {}

	for i in ind_shuffled:
		if method == "mean":
			score1_new[ind_new] = np.array([score1[x] for x in i]).mean()
			score2_new[ind_new] = np.array([score2[x] for x in i]).mean()
			score_diff_new[ind_new] = np.array([score_diff[x] for x in i]).mean()
		if method == "median":
			score1_new[ind_new] = np.median(np.array([score1[x] for x in i]))
			score2_new[ind_new] = np.median(np.array([score2[x] for x in i]))
			score_diff_new[ind_new] = np.median(np.array([score_diff[x] for x in i]))
		ind_new+=1


	# plot score1_new
	x = list(score1_new.values())
	plt.figure()
	plt.hist(x, bins=np.linspace(np.min(x),np.max(x), 50))
	plt.axvline(np.array(x).mean(), color='b', linestyle='--', linewidth=1, label='mean')
	plt.axvline(np.median(np.array(x)), color='r', linestyle='-.', linewidth=1, label='median')
	plt.legend(loc='upper right')
	plt.xlabel("Score")
	plt.ylabel("Frequency")
	plt.title("Histogram of score1 (EUs)")

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	plt.savefig(output_dir+'/hist_score1_EUs.svg')

	# plot score2_new
	y = list(score2_new.values())
	plt.figure()
	plt.hist(y, bins=np.linspace(np.min(y),np.max(y), 50))
	plt.axvline(np.array(y).mean(), color='b', linestyle='--', linewidth=1, label='mean')
	plt.axvline(np.median(np.array(y)), color='r', linestyle='-.', linewidth=1, label='median')
	plt.legend(loc='upper right')
	plt.xlabel("Score")
	plt.ylabel("Frequency")
	plt.title("Histogram of score2 (EUs)")

	plt.savefig(output_dir+'/hist_score2_EUs.svg')

	# plot score_diff_new
	z = list(score_diff_new.values())
	plt.figure()
	plt.hist(z, bins=np.linspace(np.min(z),np.max(z), 50))
	plt.axvline(np.array(z).mean(), color='b', linestyle='--', linewidth=1, label='mean')
	plt.axvline(np.median(np.array(z)), color='r', linestyle='-.', linewidth=1, label='median')
	plt.legend(loc='upper right')
	plt.xlabel("Score")
	plt.ylabel("Frequency")
	plt.title("Histogram of Score Difference (EUs)")

	plt.savefig(output_dir+'/hist_score_diff_EUs.svg')

	return([score1_new, score2_new, score_diff_new, ind_shuffled])



def random_sampling(score_diff, MAX_num_of_sample):
	"""
	This function down-samples the original score difference if the size is greater than the specified MAX sample size
	The down-sampling is done by taking the ratio between the max sample size and sample size of original score difference

	@param score_diff: score difference, dict
	@param MAX_num_of_sample: maximal sample size
	@return new_score_diff: down-sampled new score difference, dict
	@return choice_vec: a vector of 0 and 1 indicating whether the original score difference is selected or not
	"""
	z = score_diff.values()
	n = len(z)
	down_sampling_ratio = MAX_num_of_sample/float(n)

	choice_vec = np.random.choice([0,1], size=n, replace=True, p=[1-down_sampling_ratio,down_sampling_ratio])

	new_score_diff = {}
	ind = 0
	for i in score_diff.keys():
		if choice_vec[i] == 1:
			new_score_diff[ind] = score_diff[i]
			ind+=1
	return([new_score_diff, choice_vec])


def normality_test(score, alpha):
	"""
	This function invokes the Shapiro-Wilks normality test to test whether
	the input score is normally distributed.

	@param score: input score, a dictionary
	@param alpha: significance level
	@return: whether to reject to not reject, a boolean value 
			(True: canoot reject; False: reject)
	"""
	if isinstance(alpha,float) == False:
		sys.stderr.write("Invalid input alpha value! Procedure terminated at normality test step.\n")
		return
	else:
		if alpha>1 or alpha<0:
			sys.stderr.write("Invalid input alpha value! Procedure terminated at normality test step.\n")
			return
	x = list(score.values())
	norm_test = stats.shapiro(x)
	if norm_test[1]>alpha:
		return(True)
	else:
		return(False)

def skew_test(score):
	"""
	This function is to check whether the skewness of the input score indicates
	asymmetry of the underlying distribution. This is a rule-based decision to recommend
	whether the test should be based on the mean or median to measure central tendency: 
	|skewness| > 1 : highly skewed
	|skewness| in (0.5,1) : moderately skewed
	|skewness| < 0.5 : roughly symmetric

	@param score: input score
	@return: a string of "mean" or "median"
	"""
	x = list(score.values())
	skewness = stats.skew(x)

	if abs(skewness)>1:
		return([round(skewness,4),"median"])
	elif abs(skewness)<1 and abs(skewness)>0.5:
		return([round(skewness,4),"median"])
	elif abs(skewness)<0.5:
		return([round(skewness,4),"mean"])


def recommend_test(test_param, is_norm):

	list_of_tests =\
	{ 
		't' : (-1,''),
		'wilcoxon': (-1,''),
		'sign' : (-1,''),
		'bootstrap' : (-1,''),
		'permutation' : (-1,''),
		'bootstrap_med' : (-1,''),
		'permutation_med' : (-1,'')

	}

	if test_param == 'median':
		# appropriate and preferred
		list_of_tests['sign'] = (1, 'The data distribution is skewed,\
		 so median is a better measure for central tendency. Sign test is appropriate for testing for median.')
		
		# not preferred
		list_of_tests['bootstrap_med'] = (0, 'The bootstrap test based on median is appropriate for this case where the distribution is skewed, but it is computationally expensive if the sample size is large.')
		list_of_tests['permutation_med'] = (0, 'The permutation test based on median is appropriate for this case where the distribution is skewed, but it is computationally expensive if the sample size is large.')

		# not appropriate
		list_of_tests['t'] = (-1,'The student t test is not appropriate for this case since the data distribution is skewed and thus not normal.')
		list_of_tests['wilcoxon'] = (-1, 'The Wilcoxon signed rank test is not appropriate for this case since it assumes symmetric distribution around the median.')
		list_of_tests['bootstrap'] = (-1, 'The bootstrap test based on mean is not appropriate for skewed distribution because the distribution is skewed.')
		list_of_tests['permutation'] = (-1, 'The permutation test based on mean is not appropriate for skewed distribution because the distribution is skewed.')

	else:
		if is_norm:
			# appropriate and preferred
			list_of_tests['t'] = (1, 'The student t test is most appropriate for normally distributed data.')

			# not preferred
			list_of_tests['wilcoxon'] = (0, 'The Wilcoxon signed rank test is appropriate for conitnuous symmetric distributions, but t test is more powerful.')
			list_of_tests['sign'] = (0, 'The sign test is appropriate for this case, but it tests for median equality and has low statistical power.')
			list_of_tests['bootstrap'] = (0, 'The bootstrap test based on mean is appropriate for normal distribution, but it is computationally expensive and t test has higher statistical power.')
			list_of_tests['permutation'] = (0, 'The permutation test based on mean is appropriate for normal distribution, but it is computationally expensive and t test has higher statistical power.')
			list_of_tests['bootstrap_med'] = (0, 'The bootstrap test based on median is appropriate for this case, but it is computationally expensive and t test has higher statistical power.')
			list_of_tests['permutation_med'] = (0, 'The permutation test based on median is appropriate for this case, but it is computationally expensive and t test has higher statistical power.')

			# not appropriate
			# None

		else:
			# appropriate and preferred
			list_of_tests['wilcoxon'] = (1,'The Wilcoxon signed rank test is appropriate since it does not assume any specific distribution but only requires symmetry.')
			
			# not preferred
			list_of_tests['sign'] = (0, 'The sign test is appropriate for non-normal data but it has lower statistical power due to loss of information.')
			list_of_tests['bootstrap'] = (0, 'The bootstrap test based on mean is appropriate for non-normal data, but it is computationally expensive.')
			list_of_tests['permutation'] = (0, 'The permutation test based on mean is appropriate for non-normal data, but it is computationally expensive.')
			list_of_tests['bootstrap_med'] = (0, 'The bootstrap test based on median is appropriate for this case, but it is computationally expensive and Wilcoxon test has higher statistical power.')
			list_of_tests['permutation_med'] = (0, 'The permutation test based on median is appropriate for this case, but it is computationally expensive and Wilcoxon test has higher statistical power.')


			# not appropriate
			list_of_tests['t']= (-1, 'The student t test is not appropriate for this case since the data distribution is not normal.')

			
	return(list_of_tests)
