# imports
import numpy as np
import random
from scipy import stats
import itertools
from collections import defaultdict


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



def wilcoxon_r(score):
    
    def handling_ties(z_rank):
        D = defaultdict(list)
        for i,item in enumerate(z_rank):
            D[item].append(i)
        D = {k:v for k,v in D.items() if len(v)>1}
        tie_list = list(D.values())
        ties_ind = [item for sublist in tie_list for item in sublist]
        ties = [z[i] for i in ties_ind]
        return(ties)
    
    if isinstance(score,dict):
        z = list(score.values())
    else:
        z = score

    z = np.array([i for i in z if i != 0])
    
    z_rank = stats.rankdata(abs(z),method='average')

    ties = handling_ties(z_rank)

    w_p = 0
    w_m = 0
    
    for i in range(0,len(z_rank)):
        if z[i]>0:
            w_p+=z_rank[i]
        if z[i]<0:
            w_m+=z_rank[i]
    
    mu_w = len(z)*(len(z)+1)/4
    sigma_w = np.sqrt(len(z)*(len(z)+1)*(2*len(z)+1)/24-sum((np.array(ties)**3-np.array(ties))/48))
    z_score = (np.min([w_p,w_m])-mu_w)/sigma_w

    return(abs(z_score/np.sqrt(len(z))))



def hodgeslehmann(score):
	z = np.array(list(score.values()))
	z_pair = list(itertools.combinations(z, 2))

	z_pair_average = []
	for i in z_pair:
		z_pair_average.append(np.mean(i))

	return(np.median(z_pair_average))


