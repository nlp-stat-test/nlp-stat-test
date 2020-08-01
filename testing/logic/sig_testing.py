# imports
import numpy as np
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test




def run_sig_test(recommended_test, score, alpha=0.05, mu=0, B=2000):
	if isinstance(score,dict):
		x = np.array(list(score.values()))
	else:
		x = score
		
	test_stats_value = 0
	pval = 0

	# already implemented sig tests
	if recommended_test == 't':
		test_stats_value, pval = stats.ttest_1samp(x,mu)
		rejection = pval<alpha
	if recommended_test == 'wilcoxon':
		test_stats_value, pval = stats.wilcoxon(x)
		rejection = pval<alpha
	if recommended_test == 'sign':
		test_stats_value, pval = sign_test(x)
		rejection = pval<alpha

	# self implemented sig tests
	if recommended_test == 'bootstrap':
		test_stats_value, pval, rejection= bootstrap_test(x,alpha,mu,B)

	if recommended_test == 'bootstrap_med':
		test_stats_value, pval, rejection = bootstrap_test(x,alpha,mu,B,method='median')

	if recommended_test == 'permutation':
		test_stats_value, pval, rejection = permutation_test(x,alpha,B)

	if recommended_test == 'permutation_med':
		test_stats_value, pval, rejection = permutation_test(x,alpha,B,method='median')
		rejection = pval<alpha


	return((round(test_stats_value,4), round(pval,4), rejection))

def bootstrap_test(x, alpha, mu, B, method='mean'):
	if method == 'mean':
		t_ratios = []
		mu_ob = np.mean(x)
		var_ob = np.var(x,ddof=1)
		se_ob = np.sqrt(var_ob/len(x))
		for b in range(0,B):
			x_b = np.random.choice(a=x, size=len(x), replace=True)
			t_ratios.append(np.mean(x_b)/np.sqrt(np.var(x_b,ddof=1)/len(x)))
		t_ratios.sort()
		low_bound = mu_ob-np.quantile(t_ratios,1-alpha/2)*se_ob
		upp_bound = mu_ob-np.quantile(t_ratios,alpha/2)*se_ob
	elif method == 'median':
		mu_ob = np.median(x)
		t_ratios = []
		for b in range(0,B):
			z_b = np.random.choice(a=x, size=len(x), replace=True)
			t_ratios.append(np.median(z_b))
		t_ratios.sort()
		low_bound = 2*mu_ob-np.quantile(t_ratios,1-alpha/2)
		upp_bound = 2*mu_ob-np.quantile(t_ratios,alpha/2)
	
	if mu < low_bound or mu > upp_bound:
		rejection=True
	else:
		rejection=False


	return(([round(low_bound,4),round(upp_bound,4)],None,rejection))


def permutation_test(x, alpha, B, method='mean'):
	count = 0
	if method == 'mean':
		mu_ob = np.mean(x)
		for b in range(0,B):
			sign_vec = np.random.choice(np.array([-1,1]),size=len(x),replace=True)
			x_b = sign_vec*x
			mu_b = np.mean(x_b)
			if abs(mu_b)>abs(mu_ob):
				count+=1
	elif method == 'median':
		mu_ob = np.median(x)
		for b in range(0,B):
			sign_vec = np.random.choice(np.array([-1,1]),size=len(x),replace=True)
			x_b = sign_vec*x
			mu_b = np.median(x_b)
			if abs(mu_b)>abs(mu_ob):
				count+=1
	pval = float(count+1)/(B+1)
	return((mu_ob,round(pval,4),pval<alpha))

