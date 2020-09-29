# imports
import numpy as np
import os
import sys
import random
from scipy import stats

from statsmodels.stats.descriptivestats import sign_test
import matplotlib

#import sigTesting
import logic.sigTesting


matplotlib.use('Svg')
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def post_power_analysis(sig_test_name, method, score, num_of_subsample, dist_name, B, alpha, mu, output_dir, alternative="two-sided", boot_B = None):

	def get_sim_sample_sizes(z, num_of_subsample):
		partitions = np.array_split(range(len(z)), num_of_subsample)
		sample_sizes = []
		for i in partitions:
			sample_sizes.append(i[-1])
		return(sample_sizes)


	z = np.array(list(score.values()))

	sample_sizes = get_sim_sample_sizes(z, num_of_subsample)

	power_sampsizes = {}

	if method == "montecarlo": 
		if dist_name == 'normal': # currently only implement for normal dist.
			mu_hat = np.mean(z)
			var_hat = np.var(z,ddof=1)
			n = len(z)
			for i in sample_sizes:
				count = 0
				for b in range(0,B):
					z_b = np.random.normal(loc=mu_hat,scale=np.sqrt(var_hat),size=int(i))
					z_b_dict = {}
					for j in range(0,len(z_b)):
						z_b_dict[j] = z_b[j]

					(test_stats, pval, CI, rejection) = logic.sigTesting.run_sig_test("t", z_b_dict, alpha, boot_B, mu, alternative) # TO-FIX add CI
					if rejection:
						count+=1
				power_sampsizes[i] = float(count)/B
		else:
			power_sample_sizes = {}


	if method == "bootstrap":
		for i in sample_sizes:
			count = 0
			for b in range(0,B):
				z_b = np.random.choice(a = z, size = int(i), replace=True)
				z_b_dict = {}
				for j in range(0,len(z_b)):
					z_b_dict[j] = z_b[j]
				(test_stats, pval, CI, rejection) = logic.sigTesting.run_sig_test(sig_test_name, z_b_dict, alpha, boot_B, mu, alternative) # TO-FIX add CI
				if rejection:
					count+=1
			power_sampsizes[i] = float(count)/B
	
	
	# test name to display
	test_name_to_display = ''
	if sig_test_name == "t":
		test_name_to_display = 'Student t test'
	if sig_test_name == 'wilcoxon':
		test_name_to_display = "Wilcoxon signed rank test"
	if sig_test_name == "bootstrap":
		test_name_to_display = "Bootstrap test (mean)"
	if sig_test_name == "permutation":
		test_name_to_display = "Permutation test (mean)"
	if sig_test_name == "sign":
		test_name_to_display = "Sign test"
	if sig_test_name == "bootstrap_med":
		test_name_to_display = "Bootstrap test (median)"
	if sig_test_name == "permutation_med":
		test_name_to_display = "Permutation test (median)"

	x = list(power_sampsizes.keys())
	y = list(power_sampsizes.values())

	plt.figure()
	plt.plot(x,y)
	plt.xlabel("Sample Size")
	plt.ylabel("Power")
	plt.title("Power Against Different Sample Sizes for '" + test_name_to_display + "'")

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	plt.savefig(output_dir+'/power_samplesizes.svg')

	return(power_sampsizes)

