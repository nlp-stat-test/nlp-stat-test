
# imports
import numpy as np
import random
from scipy import stats


"""
This script implements functionalities for effect size estimation:
1. cohend: Cohen's d 
2. hedgesg: Hedges' g
3. wilcoxon_r: Wilcoxon r score (normal approximation)
4. hodgeslehmann: Hodges-Lehmann estimator (unbiased estimator for median)
"""


def calc_eff_size(eff_size_ind, score, alpha, B):
	"""
	this function takes the user's choice of effect size index as an input and calculates the effect size estimate

	@param: eff_size_ind, the chosen effect size index
	@param: score, the input score difference, dictionary
    @param: alpha, the significance level for CI
    @param: B, num. of iteration for bootstrap CI for hedges' g

    @return: (estimate, CI)

	"""
	if eff_size_ind == "cohend":
		return(cohend(score, alpha))

	if eff_size_ind == "hedgesg":
		d, CI = cohend(score, alpha)
		return(hedgesg(d, score, alpha, B))

	if eff_size_ind == "wilcoxonr":
		return(wilcoxon_r(score, alpha))

	if eff_size_ind == "hl":
		return(hodgeslehmann(score, alpha))



def cohend(score, alpha):
    if isinstance(score, dict):
        z = np.array(list(score.values()))
    else:
        z = np.array(score)

    d = z.mean()/np.sqrt(np.var(z,ddof=1))

    n = len(z)
    se = np.sqrt((1/n)+d**2/(2*n))
    low_bound = d-se*stats.norm.ppf(1-alpha/2)
    upp_bound = d+se*stats.norm.ppf(1-alpha/2)
    CI = (round(low_bound,5),round(upp_bound,5))

    return((round(d,5),CI))




def hedgesg(d, score, alpha, B):
    x = list(score.values())
    J = 1-(3/(4*len(x)-9))
    g = d*J

    CI = boot_BCa_CI(x, alpha, B)

    return((round(g,5),CI))


def handling_ties(z,z_rank):
    D = {}
    for i,item in enumerate(z_rank):
        if item not in D:
            D[item] = []
        D[item].append(i)
    D = {k:v for k,v in D.items() if len(v)>1}
    tie_list = list(D.values())
    ties_ind = [item for sublist in tie_list for item in sublist]
    ties = [z[i] for i in ties_ind]
    return(ties)

def wilcoxon_r(score, alpha):
    if isinstance(score,dict):
        z = list(score.values())
    else:
        z = score

    z = np.array([i for i in z if i != 0])
    
    z_rank = stats.rankdata(abs(z),method='average')

    ties = handling_ties(z,z_rank)

    w_p = 0
    w_m = 0
    
    for i in range(0,len(z_rank)):
        if z[i]>0:
            w_p+=z_rank[i]
        if z[i]<0:
            w_m+=z_rank[i]
    
    mu_w = len(z)*(len(z)+1)/4
    sigma_w = np.sqrt((len(z)*(len(z)+1)*(2*len(z)+1)-0.5*sum((np.array(ties)*(np.array(ties)+1)*(np.array(ties)-1)**3-np.array(ties))))/24)
    z_score = (w_p-mu_w)/sigma_w

    r = abs(z_score/np.sqrt(len(z)))

    CI = CI_wilcoxon(score, alpha, "two-sided")

    return((round(r,5),CI))


def hodgeslehmann(score, alpha):
    score_temp = score.copy()
    score_pair_avg = []
    for i in score.keys():
        for j in range(i,len(score_temp.keys())): #i<=j
            score_pair_avg.append(np.mean([score[i],score_temp[j]]))
    
    N = len(score.values())
    M = N*(N+1)/2
    score_pair_avg_sorted = sorted(score_pair_avg)
    t_low = np.floor(stats.norm.ppf(alpha/2)*np.sqrt(M*(N+2)/12)\
                         +(M/2))
    gamma_low = score_pair_avg_sorted[int(t_low+1)]
        
    t_upp = np.ceil(stats.norm.ppf(1-alpha/2)*np.sqrt(M*(N+2)/12)\
                        +(M/2))
    gamma_up = score_pair_avg_sorted[int(t_upp)]

    return((round(np.median(score_pair_avg),5),\
                (round(gamma_low,5),round(gamma_up,5))))
        



def get_jackknife_sample(x,i):
    
    if i > 1 and i < len(x):
        x_left = x[0:(i-1)]
        x_right = x[i+1:len(x)]

    if i == 1:
        x_left = [x[0]]
        x_right = x[(i+1):len(x)]
    
    if i == 0:
        x_left = []
        x_right = x[1:len(x)]
    
    if i == len(x):
        x_left = x[0:(len(x)-1)]
        x_right = []
    
    x_jackknife = [*x_left,*x_right]
    return(x_jackknife)


def boot_BCa_CI(x, alpha, B):
    theta_hat = []
    z_0_hat = 0
    a_hat = 0
    theta_ob = compute_g(x)
    count = 0

    
    for b in range(0,B):
        x_b = np.random.choice(x,size=len(x),replace=True)
        theta_b = compute_g(x_b)
        theta_hat.append(theta_b)
        if theta_b < theta_ob:
            count+=1
        
    # bias-correction factor estimate
    z_0_hat = stats.norm.ppf(float(count)/B)
    
    
    # acceleration factor estimate
    theta_hat_jackknife = []
    for i in range(0,len(x)):
        #print(i)
        x_jack = get_jackknife_sample(x,i)
        theta_hat_i = compute_g(x_jack)
        theta_hat_jackknife.append(theta_hat_i)
    
    theta_hat_dot = np.mean(theta_hat_jackknife)
    
    a_hat = (1/6)*sum((-np.array(theta_hat_jackknife)+theta_hat_dot)**3)/(sum((np.array(theta_hat_jackknife)-theta_hat_dot)**2))**(3/2)
    
    
    alpha1 = stats.norm.cdf(z_0_hat+\
                            ((z_0_hat+stats.norm.ppf(alpha))\
                             /(1-a_hat*(z_0_hat+stats.norm.ppf(alpha)))))
    alpha2 = stats.norm.cdf(z_0_hat+\
                           ((z_0_hat+stats.norm.ppf(1-alpha))\
                            /(1-a_hat*(z_0_hat+stats.norm.ppf(1-alpha)))))
    
    theta_hat_sorted = sorted(theta_hat)

    upp_bound = np.quantile(theta_hat_sorted,alpha2)
    low_bound = np.quantile(theta_hat_sorted,alpha1)
    return((round(low_bound,5),round(upp_bound,5)))


def compute_g(x):
    d, CI = cohend(x, 0.05)
    J = 1-(3/(4*len(x)-9))
    return(d*J)



def CI_wilcoxon(score, alpha, alternative):
    n  = len(score.values())
    M = n*(n+1)/2

    score_temp = score.copy()
    score_pair_avg = []

    for i in score.keys():
        for j in range(i,len(score_temp.keys())): #i<=j
            score_pair_avg.append(np.mean([score[i],score_temp[j]]))

    W = sorted(score_pair_avg)

    if alternative == "less":
        C_a_one = np.floor(n*(n+1)/4-stats.norm.ppf(1-alpha)*np.sqrt(n*(n+1)*(2*n+1)/24))
        upp_bound = W[int(M+1-C_a_one)]
        CI = (-float('inf'),round(upp_bound,5))

    if alternative == "greater":
        C_a_one = np.floor(n*(n+1)/4-stats.norm.ppf(1-alpha)*np.sqrt(n*(n+1)*(2*n+1)/24))
        low_bound = W[int(C_a_one)]
        CI = (round(low_bound,5), float('inf'))

    if alternative == "two-sided":
        C_a_two = np.floor(n*(n+1)/4-stats.norm.ppf(1-alpha/2)*np.sqrt(n*(n+1)*(2*n+1)/24))
        low_bound = W[int(C_a_two)]
        upp_bound = W[int(M+1-C_a_two)]
        CI = (round(low_bound,5),round(upp_bound,5))

    return(CI)

    
