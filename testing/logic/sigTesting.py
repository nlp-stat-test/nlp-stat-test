# imports
import numpy as np
from scipy import stats
import logic.effectSize




def run_sig_test(recommended_test, score, alpha, B, mu, alternative="two-sided", conf_int=False):

	if isinstance(score, dict): # check if score is a dict
		x = np.array(list(score.values()))
	else:
		x = score
		
	test_stats_value = 0.0 # initialize test statistic value, p-value and CI
	pval = 0.0
	CI = None
	rejection = None

	
	if recommended_test == 't':
		test_stats_value, pval = t_test(x, alpha, mu, alternative)
		if conf_int:
			CI = CI_mean(x, alpha, alternative)
		rejection = pval<alpha

	if recommended_test == 'wilcoxon':
		#x = x - mu
		test_stats_value, pval = wilcoxon_test(x, alpha, mu, alternative)
		if conf_int:
			CI = logic.effectSize.CI_wilcoxon(score, alpha, alternative)  #adjust for mu?
		rejection = pval<alpha

	if recommended_test == 'sign':

		test_stats_value, pval = sign_test(x, alpha, mu, alternative)
		if conf_int:
			CI = CI_median(x, alpha, alternative)
		rejection = pval<alpha

	if recommended_test == 'bootstrap':
		test_stats_value, pval, CI= bootstrap_test(x, alpha, mu, B, alternative)
		rejection = pval<alpha

	if recommended_test == 'bootstrap_med':
		test_stats_value, pval, CI = bootstrap_test(x, alpha, mu, B, alternative, method='median')
		rejection = pval<alpha

	if recommended_test == 'permutation':
		test_stats_value, pval, rejection = permutation_test(x, alpha, mu, B,alternative)
		if conf_int:
			CI = CI_mean(x, alpha, alternative)

	if recommended_test == 'permutation_med':
		test_stats_value, pval, rejection = permutation_test(x, alpha, mu, B,alternative,method='median')
		if conf_int:
			CI = CI_median(x, alpha, alternative)


	
	return((test_stats_value, pval, CI, rejection)) # to fix
	#return((test_stats_value, pval, rejection))




### confidence interval methods ###

def CI_mean(x, alpha, alternative):
	#x = x - delta # shift x by delta
	x_bar = x.mean()
	var_x = np.var(x, ddof=1)
	n = len(x)

	t = x_bar/(np.sqrt(var_x)/np.sqrt(n))

	if alternative == "less":
		upp_bound = x_bar+(np.sqrt(var_x)/np.sqrt(n))*stats.t.ppf(1-alpha,n-1)
		CI = (-float('inf'),round(upp_bound,5))

	if alternative == "greater":
		low_bound = x_bar+(np.sqrt(var_x)/np.sqrt(n))*stats.t.ppf(alpha,n-1)
		CI = (round(low_bound,5),float('inf'))

	if alternative == "two-sided":
		low_bound = x_bar+(np.sqrt(var_x)/np.sqrt(n))*stats.t.ppf(alpha/2,n-1)
		upp_bound = x_bar+(np.sqrt(var_x)/np.sqrt(n))*stats.t.ppf(1-alpha/2,n-1)
		CI = (round(low_bound,5),round(upp_bound,5))

	return(CI)


def CI_median(x, alpha, alternative):
	"""
	This function calculates the confidence interval for medians based on normal approximation

	@param x: input sample, one-dimensional array
	@param alpha: significance level
	@param alternative: the alternative hypothesis

	@return CI: confidence interval
	"""
	#x = x - delta
	x_sorted = sorted(x)
	n = len(x)
	CI = None

	if alternative == "less":
		upp_bound = x_sorted[int(round(1+n/2+stats.norm.ppf(1-alpha)*np.sqrt(n)/2))]
		CI = (-float('inf'),round(upp_bound,5))

	if alternative == "greater":
		low_bound = x_sorted[int(round(n/2-stats.norm.ppf(1-alpha)*np.sqrt(n)/2))]
		CI = (round(low_bound,5),float('inf'))

	if alternative == "two-sided":
		low_bound = x_sorted[int(round(n/2-stats.norm.ppf(1-alpha/2)*np.sqrt(n)/2))]
		upp_bound = x_sorted[int(round(1+n/2+stats.norm.ppf(1-alpha/2)*np.sqrt(n)/2))]
		CI = (round(low_bound,5),round(upp_bound,5))

	return(CI)





### sig test methods ###
def t_test(x, alpha, delta, alternative):
	"""
	This function impelements the student t test

	@param x: input sample, one-dimensional array
	@param alpha: significance level
	@param delta: null hypothesis value (mean)
	@param alternative: the alternative hypothesis

	@return (test_stats_value, pval, CI): test statistic value, confidence interval and p value
	"""

	x = x - delta # shift x by delta

	x_bar = x.mean()
	var_x = np.var(x, ddof=1)
	n = len(x)

	t = x_bar/(np.sqrt(var_x)/np.sqrt(n))

	if alternative == "less":
		pval = stats.t.cdf(t,df=n-1)

	if alternative == "greater":
		pval = 1-stats.t.cdf(t,df=n-1)

	if alternative == "two-sided":
		if t <=0:
			pval = 2*stats.t.cdf(t,df=n-1)
		if t>0:
			pval = 2*(1-stats.t.cdf(t,df=n-1))

	return((round(t,5), round(pval,5)))


def sign_test(x, alpha, delta, alternative):
	"""
	This function impelements the exaxt sign test based on binomial distribution

	@param x: input sample, one-dimensional array
	@param alpha: significance level
	@param delta: null hypothesis value (median)
	@param alternative: the alternative hypothesis

	@return (test_stats_value, pval, CI): test statistic value, confidence interval and p value
	"""

	x = x - delta # shift x by delta

	B = sum(x>0.0) # test statistic value, # of positive obs
	n = len(x)-sum(x==0.0) # sample size of non-zeros
	p = 1/2
	CI = None


	if n<20: # exact test using binomial dist
		if alternative == "greater":
			pval = stats.binom.cdf(n-B,n,p=1/2)

		if alternative == "less":
			pval = stats.binom.cdf(B,n,p=1/2)

		if alternative == "two-sided":
			if B > n/2:
				pval = 2*stats.binom.cdf(n-B,n,p=1/2)
			if B < n/2:
				pval = 2*stats.binom.cdf(B,n,p=1/2)
			if B == n/2:
				pval = 1

	if n>=20: # normal approx
		if alternative == "greater":
			pval = 1-stats.norm.cdf((B-n/2-0.5)/np.sqrt(n/4),loc=0,scale=1)

		if alternative == "less":
			pval = stats.norm.cdf((B-n/2+0.5)/np.sqrt(n/4),loc=0,scale=1)

		if alternative == "two-sided":
			if B>n/2:
				pval = 2*(1-stats.norm.cdf((B-n/2-0.5)/np.sqrt(n/4),loc=0,scale=1))
			if B<n/2:
				pval = 2*stats.norm.cdf((B-n/2+0.5)/np.sqrt(n/4),loc=0,scale=1)
			if B==n/2:
				pval = 1.0

	return((round(B,5),round(pval,5)))


def wilcoxon_test(x, alpha, delta, alternative):
	x = x - delta
	x = np.array([i for i in x if i != 0])
	n = len(x)

	x_rank = stats.rankdata(abs(x),method='average')

	ties = logic.effectSize.handling_ties(x, x_rank) #logic.

	w_p = 0
	w_m = 0

	for i in range(0,len(x_rank)):
		if x[i]>0:
			w_p+=x_rank[i]
		if x[i]<0:
			w_m+=x_rank[i]

	if w_p != n*(n+1)/4:
		if len(ties) == 0:
			T = (abs(w_p-n*(n+1)/4)-0.5)/np.sqrt(n*(n+1)*(2*n+1)/24)
		if len(ties) != 0:
			T = (abs(w_p-n*(n+1)/4)-0.5)/np.sqrt((n*(n+1)*(2*n+1)/24)-sum(np.array(ties)**3-np.array(ties))/48)
	if w_p == n*(n+1)/4:
		T = 0

	pval = 0

	if alternative == "less":
		pval = 1-stats.norm.cdf(T)

	if alternative == "greater":
		pval = stats.norm.cdf(T)

	if alternative == "two-sided":
		pval = 2*(1-stats.norm.cdf(T))

	return((round(T,5),round(pval,5)))


def bootstrap_test(x, alpha, delta, B, alternative, method='mean'):

	if method == 'mean':
		t_ratios = []
		mu_ob = np.mean(x)
		var_ob = np.var(x,ddof=1)
		se_ob = np.sqrt(var_ob/len(x))
		for b in range(0,B):
			x_b = np.random.choice(a=x, size=len(x), replace=True)
			t_ratios.append((np.mean(x_b)-mu_ob)/(np.sqrt(np.var(x_b,ddof=1)/len(x))))
		
		t_ratios.sort()

		pval = 1
		increment = 0.001

		if alternative == "less":
			upp_bound = mu_ob-np.quantile(t_ratios,alpha)*se_ob

			upp_bound_temp = mu_ob-np.quantile(t_ratios,pval)*se_ob

			while pval>increment and delta>upp_bound_temp:
				pval = pval - increment
				upp_bound_temp = mu_ob-np.quantile(t_ratios,pval)*se_ob

			CI = (-float('inf'),round(upp_bound,5))


		if alternative == "greater":
			low_bound = mu_ob-np.quantile(t_ratios,1-alpha)*se_ob

			low_bound_temp = mu_ob-np.quantile(t_ratios,1-pval)*se_ob

			while pval > increment and delta < low_bound_temp:
				pval = pval - increment
				low_bound_temp = mu_ob-np.quantile(t_ratios,pval)*se_ob

			CI = (round(low_bound,5),float('inf'))

		if alternative == "two-sided":
			low_bound = mu_ob-np.quantile(t_ratios,1-alpha/2)*se_ob
			upp_bound = mu_ob-np.quantile(t_ratios,alpha/2)*se_ob

			low_bound_temp = mu_ob-np.quantile(t_ratios,1-pval/2)*se_ob
			upp_bound_temp = mu_ob-np.quantile(t_ratios,pval/2)*se_ob

			while pval>increment and (delta < low_bound_temp or delta > upp_bound_temp):
				pval = pval - increment
				low_bound_temp = mu_ob-np.quantile(t_ratios,1-pval/2)*se_ob
				upp_bound_temp = mu_ob-np.quantile(t_ratios,pval/2)*se_ob

			CI = (round(low_bound,5),round(upp_bound,5))

	elif method == 'median':
		mu_ob = np.median(x)
		t_ratios = []
		for b in range(0,B):
			z_b = np.random.choice(a=x, size=len(x), replace=True)
			t_ratios.append(np.median(z_b))
		t_ratios.sort()


		pval = 1
		increment = 0.001

		if alternative == "less":
			upp_bound = 2*mu_ob-np.quantile(t_ratios,alpha)

			upp_bound_temp = 2*mu_ob-np.quantile(t_ratios,pval)

			while pval>increment and delta > upp_bound_temp:
				pval = pval - increment
				upp_bound_temp = 2*mu_ob-np.quantile(t_ratios,alpha)

			CI = (-float('inf'),round(upp_bound,5))


		if alternative == "greater":
			low_bound = 2*mu_ob-np.quantile(t_ratios,1-alpha)

			low_bound_temp = 2*mu_ob-np.quantile(t_ratios,1-pval)

			while pval > increment and delta < low_bound_temp:
				pval = pval - increment
				low_bound_temp = 2*mu_ob-np.quantile(t_ratios,1-pval)

			CI = (round(low_bound,5),float('inf'))

		if alternative == "two-sided":
			low_bound = 2*mu_ob-np.quantile(t_ratios,1-alpha/2)
			upp_bound = 2*mu_ob-np.quantile(t_ratios,alpha/2)

			low_bound_temp = 2*mu_ob-np.quantile(t_ratios,1-pval/2)
			upp_bound_temp = 2*mu_ob-np.quantile(t_ratios,pval/2)

			while pval>increment and (delta < low_bound_temp or delta > upp_bound_temp):
				pval = pval - increment
				low_bound_temp = 2*mu_ob-np.quantile(t_ratios,1-pval/2)
				upp_bound_temp = 2*mu_ob-np.quantile(t_ratios,pval/2)

			CI = (round(low_bound,5),round(upp_bound,5))


	return((round(mu_ob,5), round(pval,5), CI))


def permutation_test(x, alpha, delta, B, alternative, method='mean'):
	x = x - delta

	count = 0
	if method == 'mean':
		mu_ob = np.mean(x)
		for b in range(0,B):
			sign_vec = np.random.choice(np.array([-1,1]),size=len(x),replace=True)
			x_b = sign_vec*x
			mu_b = np.mean(x_b)
			if alternative == "less":
				if mu_b < mu_ob:
					count+=1
			if alternative == "greater":
				if mu_b > mu_ob:
					count+=1
			if alternative == "two-sided":
				if abs(mu_b)>abs(mu_ob):
					count+=1
	elif method == 'median':
		mu_ob = np.median(x)
		for b in range(0,B):
			sign_vec = np.random.choice(np.array([-1,1]),size=len(x),replace=True)
			x_b = sign_vec*x
			mu_b = np.median(x_b)
			if alternative == "less":
				if mu_b < mu_ob:
					count+=1
			if alternative == "greater":
				if mu_b > mu_ob:
					count+=1
			if alternative == "two-sided":
				if abs(mu_b)>abs(mu_ob):
					count+=1

	pval = float(count+1)/(B+1)
	return((round(mu_ob,5),round(pval,5),pval<alpha))

