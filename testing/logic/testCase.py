"""
This is a script containing the object testCase class.
"""
import numpy as np
class testCase:    
    def __init__(self, score1, score2, score_diff, score_diff_par, sample_size):
        # attributes
        self.score1 = score1
        self.score2 = score2
        self.score_diff = score_diff
        self.score_diff_par = score_diff_par
        self.sample_size = sample_size

        self.eda = self.eda(0, False, '', 0, 0.0, False, '', None)

        self.sigTest = self.sigTest('', 0.0, 0, 0, 0, 0, False)

        self.es = self.es([],[])

        self.power = self.power({}, '', '', 0.0, 0, 0.0, 0)



    class eda:
        def __init__(self, m, isShuffled, calc_method, randomSeed, normal_alpha, normal, testParam, testName):
            self.m = m
            self.isShuffled = isShuffled
            self.calc_method = calc_method
            self.randomSeed = randomSeed
            self.normal_alpha = normal_alpha
            self.normal = normal
            self.testParam = testParam
            self.testName = testName

            self.summaryStat_score1 = self.summaryStat(0,0,0,0,0)
            self.summaryStat_score2 = self.summaryStat(0,0,0,0,0)
            self.summaryStat_score_diff = self.summaryStat(0,0,0,0,0)
            self.summaryStat_score_diff_par = self.summaryStat(0,0,0,0,0)

        class summaryStat:
            def __init__(self, mu, sd, med, min_val, max_val):
                self.mu = mu
                self.sd = sd
                self.med = med
                self.min = min_val
                self.max = max_val

    class sigTest:
        def __init__(self, testName, alpha, B, mu, test_stat, pval, rejection):
            self.testName = testName
            self.alpha = alpha
            self.B = B
            self.mu = mu
            self.test_stat = test_stat
            self.pval = pval
            self.rejection = rejection

    class es:
        def __init__(self, estimate, estimator):
            self.estimate = estimate
            self.estimator = estimator

    class power:
        def __init__(self, powerCurve, method, dist_name, alpha, num_of_subsample, pow_lev, B):
            self.powerCurve = powerCurve
            self.method = method
            self.dist_name = dist_name
            self.alpha = alpha
            self.num_of_subsample = num_of_subsample
            self.pow_lev = pow_lev
            self.B = B



    def calc_score_diff(self):
        self.score_diff = {}
        for i in self.score1.keys():
            self.score_diff[i] = self.score1[i]-self.score2[i]

    def get_summary_stats(self):
        x = np.array(list(self.score1.values()))
        y = np.array(list(self.score2.values()))
        z = np.array(list(self.score_diff.values()))

        z_par = np.array(list(self.score_diff_par.values()))


        self.eda.summaryStat_score1.mu = np.mean(x)
        self.eda.summaryStat_score1.sd = np.sqrt(np.var(x,ddof=1))
        self.eda.summaryStat_score1.med = np.median(x)
        self.eda.summaryStat_score1.min_val = np.min(x)
        self.eda.summaryStat_score1.max_val = np.max(x)

        self.eda.summaryStat_score2.mu = np.mean(y)
        self.eda.summaryStat_score2.sd = np.sqrt(np.var(y,ddof=1))
        self.eda.summaryStat_score2.med = np.median(y)
        self.eda.summaryStat_score2.min_val = np.min(y)
        self.eda.summaryStat_score2.max_val = np.max(y)

        self.eda.summaryStat_score_diff.mu = np.mean(z)
        self.eda.summaryStat_score_diff.sd = np.sqrt(np.var(z,ddof=1))
        self.eda.summaryStat_score_diff.med = np.median(z)
        self.eda.summaryStat_score_diff.min_val = np.min(z)
        self.eda.summaryStat_score_diff.max_val = np.max(z)

        self.eda.summaryStat_score_diff_par.mu = np.mean(z_par)
        self.eda.summaryStat_score_diff_par.sd = np.sqrt(np.var(z_par,ddof=1))
        self.eda.summaryStat_score_diff_par.med = np.median(z_par)
        self.eda.summaryStat_score_diff_par.min_val = np.min(z_par)
        self.eda.summaryStat_score_diff_par.max_val = np.max(z_par)







