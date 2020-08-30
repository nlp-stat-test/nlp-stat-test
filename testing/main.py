import logic.fileReader
import logic.dataAnalysis
import logic.sigTesting
import logic.testCase
import logic.effectSize
import logic.powerAnalysis

import sys
import time
import numpy as np

import yaml


if __name__ == '__main__':

	score_file = sys.argv[1] # load score file

	config_file = sys.argv[2] # load config file
	with open(config_file, 'r') as ymlfile:
		config = yaml.load(ymlfile)

	with open('logic/sysconfig.yml', 'r') as ymlfile:
		sysconfig = yaml.load(ymlfile)


	### initialize a new testCase object
	testCase_new = logic.testCase.testCase(None, None, None, None, None)

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
	es_alpha = float(config['eff_size_alpha'])
	es_B = int(sysconfig['NUM_of_iter_boot'])

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
	mu = float(config['mu'])

	# h1
	h1 = str(config['h1'])


	### read score file
	[testCase_new.score1,testCase_new.score2] = logic.fileReader.read_score_file(score_file)



	## data analysis
	# plot histograms
	print("------ EDA ------")

	# calculate score difference
	testCase_new.calc_score_diff()

	testCase_new.sample_size = np.floor(len(list(testCase_new.score1.values()))/float(testCase_new.eda.m))

	# partition score difference and plot hists
	testCase_new.score1, testCase_new.score2, testCase_new.score_diff_par, ind_shuffled = logic.dataAnalysis.partition_score(\
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
	#MIN_power_samplesize = sysconfig['MIN_power_samplesize']
	#if MIN_power_samplesize>float(testCase_new.sample_size)/testCase_new.power.num_of_subsample:
		#print("Sample size too small for power analysis simulation for certain significance tests. Decrease the number of simulations.")
		#sys.exit()




	# summary statistics

	testCase_new.get_summary_stats()

	# skewness test
	testCase_new.eda.testParam = logic.dataAnalysis.skew_test(testCase_new.score_diff_par)[1]


	# normality test
	testCase_new.eda.normal = logic.dataAnalysis.normality_test(\
		score = testCase_new.score_diff_par, 
		alpha = testCase_new.eda.normal_alpha)


	# recommend tests
	list_of_tests = logic.dataAnalysis.recommend_test(testCase_new.eda.testParam,testCase_new.eda.normal)

	recommended_test_list = []
	not_preferred_test_list = []
	inappro_test_list = []

	for i,j in list_of_tests.items():
		if j[0] == 1:
			recommended_test_list.append((i,j[1]))
		if j[0] == 0:
			not_preferred_test_list.append((i,j[1]))
		if j[0] == -1:
			inappro_test_list.append((i,j[1]))


	print('Appropriate and recommended tests are:')
	for i in recommended_test_list:
		print(str(i[0])+': '+str(i[1]))
	print('--------------------')

	print('Appropriate but not preferred tests are:')
	for i in not_preferred_test_list:
		print(str(i[0])+': '+str(i[1]))
	print("====================")

	print("Inappropriate tests are:")
	for i in inappro_test_list:
		print(str(i[0])+': '+str(i[1]))
	print("====================")


	chosen_test = input("Please choose a test:\n")

	for i in inappro_test_list:
		if i[0] == chosen_test:
			chosen_test = input("Please choose from the list of recommended tests:\n")

	testCase_new.sigTest.testName = chosen_test


	print("====================")

	print('Sample size after partitioning is: '+str(testCase_new.sample_size))
	print('Significance test chosen: '+testCase_new.sigTest.testName)
	print('Normality: '+str(testCase_new.eda.normal))
	print('Testing parameter: '+testCase_new.eda.testParam)

	# run sig test
	print("------ Testing ------")
	test_stat, pval, CI, rejection = logic.sigTesting.run_sig_test(\
		recommended_test = testCase_new.sigTest.testName, 
		score = testCase_new.score_diff_par, 
		alpha = testCase_new.sigTest.alpha, 
		B = testCase_new.sigTest.B, 
		mu=mu,
		alternative = h1,
		conf_int = True)

	testCase_new.sigTest.test_stat = test_stat
	testCase_new.sigTest.pval = pval
	testCase_new.sigTest.rejection = rejection

	### effect size calculation

	print("------ Effect Size ------")

	eff_size_est, es_CI = logic.effectSize.calc_eff_size(testCase_new.es.estimator, testCase_new.score_diff_par, es_alpha, es_B)

	testCase_new.es.estimate = eff_size_est

	### post power analysis

	print("------ Power Analysis ------")
	start_time = time.time()

	power_sampsize = logic.powerAnalysis.post_power_analysis(\
		sig_test_name = testCase_new.sigTest.testName,
		method = testCase_new.power.method,
		score = testCase_new.score_diff_par, 
		num_of_subsample= testCase_new.power.num_of_subsample, 
		dist_name = testCase_new.power.dist_name, 
		B = testCase_new.power.B,
		alpha = testCase_new.power.alpha,
		mu = mu,
		output_dir = fig_output_dir,
		alternative = h1,
		boot_B = testCase_new.sigTest.B)

	sys.stderr.write("Finished power analysis. Runtime: --- %s seconds ---" % (time.time() - start_time) + '\n')
    
	testCase_new.power.powerCurve = power_sampsize


	print('------ Report ------')

	print('Test name: '+testCase_new.sigTest.testName)
	print('Confidence interval: '+str(CI))
	print('P-value: '+str(testCase_new.sigTest.pval))
	print('Rejection of H0: '+str(testCase_new.sigTest.rejection))


	print('-----------')

	print('effect size estimates: '+str(testCase_new.es.estimate))
	print('effect size estimator: '+str(testCase_new.es.estimator))
	print('Confidence interval for effect size: '+str(es_CI))

	print('-----------')

	print('obtained power: ' + str(list(testCase_new.power.powerCurve.values())[-1]))
	print('Power analysis method: '+str(testCase_new.power.method))






