
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


def calc_eff_size(eff_size_ind, score):
	"""
	this function takes the user's choice of effect size index as an input and calculates the effect size estimate

	@param: eff_size_ind, the chosen effect size index
	@param: score, the input score difference, dictionary

	"""
	if eff_size_ind == "cohend":
		return(round(cohend(score),4))

	if eff_size_ind == "hedgesg":
		d = cohend(score)
		return(round(hedgesg(d,score),4))

	if eff_size_ind == "wilcoxonr":
		return(round(wilcoxon_r(score),4))

	if eff_size_ind == "hl":
		return(round(hodgeslehmann(score),4))



def cohend(score):
	z = np.array(list(score.values()))

	return(z.mean()/np.sqrt(np.var(z,ddof=1)))

def hedgesg(d,score):
	J = 1-(3/(4*len(list(score.values()))-9))
	return(d*J)


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

def wilcoxon_r(score):
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

    return(abs(z_score/np.sqrt(len(z))))


def hodgeslehmann(score, conf_int = False, conf_lev = 0.05):
    score_temp = score.copy()
    score_pair_avg = []
    for i in score.keys():
        for j in range(i,len(score_temp.keys())): #i<=j
            score_pair_avg.append(np.mean([score[i],score_temp[j]]))
            
    if conf_int:
        N = len(score.values())
        M = N*(N+1)/2
        score_pair_avg_sorted = sorted(score_pair_avg)
        t_low = np.floor(stats.norm.ppf(conf_lev/2)*np.sqrt(M*(N+2)/12)\
                         +(M/2))
        gamma_low = score_pair_avg_sorted[int(t_low+1)]
        
        t_upp = np.ceil(stats.norm.ppf(1-conf_lev/2)*np.sqrt(M*(N+2)/12)\
                        +(M/2))
        gamma_up = score_pair_avg_sorted[int(t_upp)]
        return((round(np.median(score_pair_avg),4),\
                (round(gamma_low,4),round(gamma_up,4))))
        
        
    #return((round(np.median(score_pair_avg),4),None))
    return(round(np.median(score_pair_avg),4)) # TO-FIX


