import fileReader
import dataAnalysis
import sigTesting
import testCase
import effectSize
import powerAnalysis

import sys
import time
import numpy as np

import yaml


if __name__ == '__main__':

	score_file = sys.argv[1] # load score file

	config_file = sys.argv[2] # load config file
	with open(config_file, 'r') as ymlfile:
		config = yaml.load(ymlfile)

	with open('sysconfig.yml', 'r') as ymlfile:
		sysconfig = yaml.load(ymlfile)


	### initialize a new testCase object
	testCase_new = testCase.testCase(None, None, None, None, None)

	# eda
	testCase_new.eda.m = int(config['eval_unit_size'])
	testCase_new.eda.calc_method = str(config['eval_unit_met'])
	testCase_new.eda.isShuffled = bool(config['shuffled'])
	testCase_new.eda.randomSeed = int(config['random_seed'])
	testCase_new.eda.normal_alpha = float(config['normal_test_alpha'])

	# sig test
	testCase_new.sigTest.alpha = float(config['sig_test_alpha'])
	testCase_new.sigTest.B = int(config['sig_boot_perm_B'])

	# effect size
	testCase_new.es.estimator = str(config['eff_size_ind'])

	# power analysis
	testCase_new.power.alpha = float(config['power_alpha'])
	testCase_new.power.method = str(config['power_method'])
	testCase_new.power.dist_name = str(config['power_dist_name'])
	testCase_new.power.num_of_subsample = int(config['power_num_of_subsample'])
	testCase_new.power.B = int(config['power_B'])


	# output dir
	fig_output_dir = str(config['fig_output_dir'])
	#report_output_dir = str(config['report_output_dir'])

	# null hypothesis
	mu = float(sysconfig['mu'])


	### read score file
	[testCase_new.score1,testCase_new.score2] = fileReader.read_score_file(score_file)



	## data analysis
	# plot histograms
	print("------ EDA ------")
	#dataAnalysis.plot_hist(testCase_new.score1, testCase_new.score2, 'figures')

	# calculate score difference
	testCase_new.calc_score_diff()

	testCase_new.sample_size = np.floor(len(list(testCase_new.score1.values()))/float(testCase_new.eda.m))

	# partition score difference and plot hists
	testCase_new.score1, testCase_new.score2, testCase_new.score_diff_par, ind_shuffled = dataAnalysis.partition_score(\
		score1 = testCase_new.score1, 
		score2 = testCase_new.score2, 
		score_diff = testCase_new.score_diff, 
		eval_unit_size = testCase_new.eda.m, 
		shuffled = testCase_new.eda.isShuffled, 
		randomSeed = testCase_new.eda.randomSeed, 
		method = testCase_new.eda.calc_method,
		output_dir = fig_output_dir)


	# check for minimum sample size requirement for power analysis
	# this check is here because wilcoxon test needs more than 10 data points
	MIN_power_samplesize = sysconfig['MIN_power_samplesize']
	if MIN_power_samplesize>float(testCase_new.sample_size)/testCase_new.power.num_of_subsample:
		print("Sample size too small for power analysis simulation for certain significance tests. Decrease the number of simulations.")
		sys.exit()




	# summary statistics

	testCase_new.get_summary_stats()

	# skewness test
	testCase_new.eda.testParam = dataAnalysis.skew_test(testCase_new.score_diff_par)


	# normality test
	testCase_new.eda.normal = dataAnalysis.normality_test(\
		score = testCase_new.score_diff_par, 
		alpha = testCase_new.eda.normal_alpha)


	# recommend tests
	recommended_tests = dataAnalysis.recommend_test(testCase_new.eda.testParam,testCase_new.eda.normal)


	print('Sample size after partitioning is: '+str(testCase_new.sample_size))
	print('the test used: '+testCase_new.sigTest.testName)
	print('normality: '+str(testCase_new.eda.normal))
	print('testing parameter: '+testCase_new.eda.testParam)

	print('Recommended tests are:')
	for i in recommended_tests:
		print("test name: "+str(i[0]))
		print("----- reason: "+str(i[1]))
		print("=============")

	testCase_new.sigTest.testName= input("Please choose a test:\n")
 
	testCase_new.eda.testName = testCase_new.sigTest.testName


	# run sig test
	print("------ Testing ------")
	test_stat, pval, rejection = sigTesting.run_sig_test(\
		recommended_test = testCase_new.sigTest.testName, 
		score = testCase_new.score_diff_par, 
		alpha = testCase_new.sigTest.alpha, 
		B = testCase_new.sigTest.B, 
		mu=mu)

	testCase_new.sigTest.test_stat = test_stat
	testCase_new.sigTest.pval = pval
	testCase_new.sigTest.rejection = rejection

	### effect size calculation

	print("------ Effect Size ------")

	eff_size_est = effectSize.calc_eff_size(testCase_new.es.estimator, testCase_new.score_diff_par)

	testCase_new.es.estimate = eff_size_est

	### post power analysis

	print("------ Power Analysis ------")
	start_time = time.time()

	power_sampsize = powerAnalysis.post_power_analysis(\
		sig_test_name = testCase_new.sigTest.testName,
		method = testCase_new.power.method,
		score = testCase_new.score_diff_par, 
		num_of_subsample= testCase_new.power.num_of_subsample, 
		dist_name = testCase_new.power.dist_name, 
		B = testCase_new.power.B,
		alpha = testCase_new.power.alpha,
		mu = mu,
		output_dir = fig_output_dir,
		boot_B = testCase_new.sigTest.B)

	sys.stderr.write("Finished power analysis. Runtime: --- %s seconds ---" % (time.time() - start_time) + '\n')
    
	testCase_new.power.powerCurve = power_sampsize


	print('------ Report ------')

	print('test name: '+testCase_new.sigTest.testName)
	print('test statistic/CI: '+str(testCase_new.sigTest.test_stat))
	print('p-value: '+str(testCase_new.sigTest.pval))
	print('rejection of H0: '+str(testCase_new.sigTest.rejection))


	print('-----------')

	print('effect size estimates: '+str(testCase_new.es.estimate))
	print('effect size estimator: '+str(testCase_new.es.estimator))

	print('-----------')

	print('obtained power: ' + str(list(testCase_new.power.powerCurve.values())[-1]))
	print('Power analysis method: '+str(testCase_new.power.method))







