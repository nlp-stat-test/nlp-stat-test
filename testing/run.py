# v3
import io

import yaml
from flask import *
from flask import render_template

from werkzeug.utils import secure_filename
import os
import numpy as np

# Business Logic
from logic.fileReader import read_score_file
from logic.testCase import testCase
from logic.helper import helper
from logic.effectSize import calc_eff_size
from logic.dataAnalysis import partition_score, \
    skew_test, normality_test, recommend_test
from logic.sigTesting import run_sig_test
from logic.power_analysis_norm import prosp_power_analysis_norm

# filenames
# from logic.filenames import get_path

# Report Function
from logic.report import gen_report

from logic.errorHandling import InputError

FOLDER = os.path.join('user')
from logic.powerAnalysis import post_power_analysis
import logic.powerAnalysis

app = Flask(__name__)
app.config['FOLDER'] = FOLDER

# defaults
DEFAULT_SEED = None
DEFAULT_EVAL_SIZE = 1

# template filename
# Note: "tab_inteface2.html" has histograms before recommendations
template_filename = "tab_interface.html"

# strings to use in UI
summary_str = "Summary of Statistics"
teststat_heading = "Test Statistic Recommendation"
sig_test_heading = 'Recommended Significance Tests'
estimators = {"cohend": "This function calculates the Cohen's d effect size estimator.",
              "hedgesg": "This function takes the Cohen's d estimate as an input and calculates the Hedges's g.",
              "wilcoxonr": "This function calculates the standardized z-score (r) for the Wilcoxon signed-rank test.",
              "hl": "This function estimates the Hodges-Lehmann estimator for the input score."}

def get_rand_state_str():
    '''
    This is a random string that's generate every time a 'submit' operation happens.
    It's used to distinguish when the state of the app may have changed.
    @return: A random int between 0 and 9999
    '''
    rand_str = str(np.random.randint(10000))
    return rand_str

def calc_score_diff(score1, score2):
    """
    This function calculates the pairwise score difference for score1 and score2

    @param score1, score2: input scores, dictionary
    @return: score_diff, score difference, a dictionary
    """
    score_diff = {}
    for i in score1.keys():
        score_diff[i] = score1[i] - score2[i]
    return (score_diff)


def partition_score_no_hist(score1, score2, score_diff, eval_unit_size, shuffled,
                            randomSeed, method):
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
    ind = list(score_diff.keys())  # the keys should be the same for three scores

    if shuffled:
        ind_shuffled = np.random.Random(randomSeed).shuffle(ind)
    print(eval_unit_size)
    ind_shuffled = np.array_split(ind, np.floor(len(ind) / eval_unit_size))
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
        ind_new += 1

    return ([score1_new, score2_new, score_diff_new, ind_shuffled])


def create_test_reasons(recommended_tests, test_statistic='mean'):
    '''
    This function creates a dictionary of test names with reasons, given the list of test names.
    @param recommended_tests: List of tuples [('t', "t because..."), ('bootstrap', 'bootstrap because...')]
    @return: Dictionary of test names with reasons as the values
    '''
    #test_reasons = {}
    #for test in recommended_tests:  # test is a tuple (name, reason)
    #    test_reasons[test[0]] = test[1]
    #return test_reasons

    # sort based on reverse order
    recommended_tests = {k: v for k, v in sorted(recommended_tests.items(),
                                                 reverse=True,
                                                 key=lambda item: item[1])}
    recommended_list = []
    not_preferred_list = []
    not_recommended_list = []
    for k, v in recommended_tests.items():
            if v[0] > 0:
                recommended_list.append((k, v[1]))
            elif v[0] == 0:
                not_preferred_list.append((k, v[1]))
            else:
                not_recommended_list.append((k, v[1]))
    return recommended_list, not_preferred_list, not_recommended_list


def format_digits(num, sig_digits=5):
    str = '{:.5f}'.format(num)
    return str


def create_summary_stats_list(tc, debug=False):
    if debug: print('Score 1: mean={}, med={}, sd={}, min={}, max={}'.format(tc.eda.summaryStat_score1.mu,
                                                                             tc.eda.summaryStat_score1.med,
                                                                             tc.eda.summaryStat_score1.sd,
                                                                             tc.eda.summaryStat_score1.min_val,
                                                                             tc.eda.summaryStat_score1.max_val))
    summary_dict = []
    # 'score1'
    summary_dict.append(('score1', [('mean', format_digits(tc.eda.summaryStat_score1.mu)),
                                    ('median', format_digits(tc.eda.summaryStat_score1.med)),
                                    ('std.dev.', format_digits(tc.eda.summaryStat_score1.sd)),
                                    ('min', format_digits(tc.eda.summaryStat_score1.min_val)),
                                    ('max', format_digits(tc.eda.summaryStat_score1.max_val))]))
    summary_dict.append(('score2', [('mean', format_digits(tc.eda.summaryStat_score2.mu)),
                                     ('median', format_digits(tc.eda.summaryStat_score2.med)),
                                    ('std.dev.', format_digits(tc.eda.summaryStat_score2.sd)),
                                    ('min', format_digits(tc.eda.summaryStat_score2.min_val)),
                                    ('max', format_digits(tc.eda.summaryStat_score2.max_val))]))
    summary_dict.append(('difference', [('mean', format_digits(tc.eda.summaryStat_score_diff_par.mu)),
                                        ('median', format_digits(tc.eda.summaryStat_score_diff_par.med)),
                                        ('std.dev.', format_digits(tc.eda.summaryStat_score_diff_par.sd)),
                                        ('min', format_digits(tc.eda.summaryStat_score_diff_par.min_val)),
                                        ('max', format_digits(tc.eda.summaryStat_score_diff_par.max_val))]))
    return summary_dict


@app.route('/', methods=["GET", "POST"])
def data_analysis(debug=True):
    if request.method == 'POST':
        # ------- Test if 'last_tab' was sent
        last_tab_clicked = request.form.get('last_tab')
        # todo: make this a cookie
        last_tab_name_clicked = 'Data Analysis'  # request.form.get('last_tab_name')
        print("***** LAST TAB: {}".format(last_tab_clicked))
        print("***** LAST TAB: {}".format(last_tab_name_clicked))

        eval_unit_size = request.form.get('eval_unit_size')

        # Handle case of no eval unit size
        if not eval_unit_size:
            eval_unit_size = DEFAULT_EVAL_SIZE

        # target_stat is 'mean' or 'median'
        eval_unit_stat = request.form.get('target_statistic')
        print('eval_unit_stat={}'.format(eval_unit_stat))

        # normality
        normality_alpha = float(request.form.get('normality_alpha'))
        print('NORMALITY_ALPHA (from form)={}'.format(normality_alpha))
        seed = request.form.get('seed')
        if not seed:
            shuffle = False
            # seed = DEFAULT_SEED
        else:
            shuffle = True

        # ------- File ----------------
        f = request.files['data_file']  # new
        have_file = False
        have_filename = False
        have_data = False
        if f.filename:
            have_filename = True
            data_filename = f.filename
            f.save(FOLDER + "/" + secure_filename(data_filename))
            have_file = True  # assume the above worked
        elif request.cookies.get('fileName'):
            data_filename = request.cookies.get('fileName')
            have_filename = True
        else:
            # no filename, print error message
            str_err = 'You must submit a file.'
            print('ERROR: submitted without filename! You must resubmit!')

        if have_filename:
            if debug: print('have filename:{}'.format(data_filename))
            try:
                # todo: raise InputException if scores1 or scores2 empty due to bad file
                scores1, scores2 = read_score_file(FOLDER + "/" + data_filename)
                # if not scores1 or not scores2:
                #     raise InputError('linewitherror','each line must be two values separated by whitespace')

                if len(scores1) > 0 and len(scores2):
                    score_dif = calc_score_diff(scores1, scores2)
                    # todo: display how many samples there are
                    num_eval_units = int(np.floor(len(list(score_dif)) / float(eval_unit_size)))
                    if debug: print('SAMPLE SIZE (#eval units)={}'.format(num_eval_units))
                    have_data = True
            except InputError as e:
                print('Exception: InputError. Line={}. {}'.format(e.line_num, e.message))
                str_err = 'Error in line {} of file {}. {}'.format(e.line_num, data_filename, e.message)
            except:
                # Todo: Can we print sys.stderr message here?
                print('Exception occurred reading file: filename={}'.format(data_filename))
                str_err = 'Exception occurred reading file: filename={}'.format(data_filename)

        if have_data:
            # partition score difference and save svg
            score_diff_par = partition_score(scores1, scores2, score_dif, float(eval_unit_size),
                                             shuffle,  # shuffle if we have seed
                                             seed,
                                             eval_unit_stat,  # mean or median
                                             FOLDER)

            # --------------Summary Stats -------------
            ### initialize a new testCase object to use for summary statistics
            tc = testCase(score_diff_par[0],
                          score_diff_par[1],
                          score_dif,
                          score_diff_par[2],  # score_diff_par,
                          num_eval_units)
            tc.get_summary_stats()

            summary_stats_list = create_summary_stats_list(tc)

            # --------------Recommended Test Statistic (mean or median, by skewness test) ------------------
            mean_or_median = skew_test(score_diff_par[2])[1]
            skewness_gamma = skew_test(score_diff_par[2])[0]

            # ---------------normality test
            is_normal = normality_test(score_diff_par[2], alpha=normality_alpha)
            # --------------Recommended Significance Tests -------------------------
            recommended_tests = recommend_test(mean_or_median, is_normal)

            # create_test_reasons returns a list of 3
            ( recommended_tests, not_preferred_tests, not_recommended_tests) = \
                create_test_reasons(recommended_tests)

            if debug:
                print("Recommended: {}".format(recommended_tests))
                print("Appropriate (not preferred): {}".format(not_preferred_tests))
                print("Inappropriate: {}".format(not_recommended_tests))

            USE_JSON = False
            if USE_JSON:
                return jsonify(result=sig_test_heading,
                               hist_score1_file='hist_score1_partitioned.svg',
                               hist_score2_file='hist_score2_partitioned.svg')
            else:
                rand = np.random.randint(10000)
                if debug: print('random number to append to image url={}'.format(rand))

                rendered = render_template(template_filename,
                                           normality_alpha=normality_alpha,
                                           skewness_gamma=skewness_gamma,
                                           hist_score1_file='hist_score1_partitioned.svg',
                                           hist_score2_file='hist_score2_partitioned.svg',
                                           hist_diff_file='hist_score_diff.svg',
                                           hist_diff_par_file='hist_score_diff_partitioned.svg',
                                           file_uploaded="File selected: {}".format(data_filename),
                                           last_tab_name_clicked=last_tab_name_clicked,
                                           eval_unit_size=eval_unit_size,
                                           eval_unit_stat=eval_unit_stat,
                                           num_eval_units=num_eval_units,
                                           shuffle_seed=seed,
                                           sig_test_heading=sig_test_heading,
                                           summary_str=summary_str,
                                           summary_stats_list=summary_stats_list,
                                           teststat_heading=teststat_heading,
                                           sigtest_heading=sig_test_heading,
                                           mean_or_median=mean_or_median,  # 'mean' if not skewed, 'median' if skewed.
                                           is_normal=is_normal,  # True if normal, False if not.
                                           recommended_tests=recommended_tests,  # list of tuples
                                           not_recommended_tests=not_recommended_tests, # list of tuples
                                           not_preferred_tests=not_preferred_tests,  # list of tuples
                                           rand=rand,  # rand is for image URL to force reload (avoid caching)
                                           # specific to effect size test
                                           effect_size_estimators=estimators,
                                           # power
                                           power_path=request.cookies.get('power_path'),
                                           power_test=request.cookies.get('power_test'),
                                           power_num_intervals=request.cookies.get('power_num_intervals'),
                                           rand_str=get_rand_state_str()
                                           )
                resp = make_response(rendered)

                # -------------- Set all cookies -------------
                if f.filename:
                    resp.set_cookie('fileName', f.filename)
                resp.set_cookie('normality_alpha', json.dumps(normality_alpha))
                resp.set_cookie('skewness_gamma', json.dumps(skewness_gamma))
                resp.set_cookie('eval_unit_size', eval_unit_size)
                resp.set_cookie('eval_unit_stat', eval_unit_stat)
                resp.set_cookie('num_eval_units', str(num_eval_units))
                resp.set_cookie('shuffle_seed', seed)

                resp.set_cookie('summary_str', summary_str)
                serialized_summary_stats_list = json.dumps(summary_stats_list)
                resp.set_cookie('summary_stats_list', serialized_summary_stats_list)

                resp.set_cookie('teststat_heading', teststat_heading)
                resp.set_cookie('mean_or_median', mean_or_median)
                print('DA Set cookie: is_normal dumped={}'.format(json.dumps(is_normal)))
                resp.set_cookie('is_normal', json.dumps(is_normal))

                resp.set_cookie('sig_test_heading', sig_test_heading)
                resp.set_cookie('recommended_tests', json.dumps(recommended_tests))
                resp.set_cookie('not_recommended_tests', json.dumps(not_recommended_tests))
                resp.set_cookie('not_preferred_tests', json.dumps(not_preferred_tests))

                resp.set_cookie('hist_score1_file', 'hist_score1_partitioned.svg')
                resp.set_cookie('hist_score2_file', 'hist_score2_partitioned.svg')
                resp.set_cookie('hist_diff_file', 'hist_score_diff.svg')
                resp.set_cookie('hist_diff_par_file', 'hist_score_diff_partitioned.svg')
                return resp  # return rendered
        else:
            # no file
            rendered = render_template(template_filename,
                                       rand_str=get_rand_state_str(),
                                       error_str=str_err, )
            return rendered
    elif request.method == 'GET':
        # You got to the main page by navigating to the URL, not by clicking submit
        return render_template(template_filename,
                               tooltip_read_score_file=helper("read_score_file"),
                               tooltip_plot_hist=helper("plot_hist"),
                               tooltip_plot_hist_diff=helper("plot_hist_diff"),
                               tooltip_partition_score=helper("partition_score"),
                               tooltip_normality_test=helper("normality_test"),
                               tooltip_skew_test=helper("skew_test"),
                               tooltip_recommend_test=helper("recommend_test"),
                               tooltip_calc_eff_size=helper("calc_eff_size"),
                               tooltip_cohend=helper("cohend"),
                               tooltip_hedgesg=helper("hedgesg"),
                               tooltip_wilcoxon_r=helper("wilcoxon_r"),
                               tooltip_hodgeslehmann=helper("hodgeslehmann"),
                               tooltip_run_sig_test=helper("run_sig_test"),
                               tooltip_bootstrap_test=helper("bootstrap_test"),
                               tooltip_permutation_test=helper("permutation_test"),
                               tooltip_post_power_analysis=helper("post_power_analysis"),
                               file_uploaded="Upload a file.",
                               recommended_tests=[],
                               summary_stats_list={},
                               rand_str=get_rand_state_str()
                               )

# ************************************************************
#   PROSPECTIVE POWER
# ************************************************************
@app.route('/prospective_power', methods=["GET", "POST"])
def prospective_power(debug=True):
    if request.method == "POST":
        # ------- Get cookies
        fileName = request.cookies.get('fileName')
        # ------- Get form data
        prospective_mu = request.form.get('prospective_mu')
        prospective_sig_alpha = request.form.get('prospective_signifcance_level')
        prospective_stddev = request.form.get('prospective_stddev')
        prospective_alternative = request.form.get('prospective_alternative')
        prospective_desired_power = request.form.get('prospective_desired_power')
        # data analysis
        show_non_recommended = request.form.get('checkbox_show_non_recommended')
        show_non_preferred = request.form.get('checkbox_show_non_preferred')
        sig_test_name = request.form.get('target_sig_test')
        sig_alpha = request.form.get('significance_level')
        # mu = float(request.form.get('mu'))
        # if not mu:
        #     mu = 0.0


        # ------- Test if 'last_tab' was sent
        last_tab_name_clicked = 'Prospective power'  # request.form.get('last_tab_input')
        if debug: print("***** LAST TAB (from POST): {}".format(last_tab_name_clicked))

        prospective_required_sample = prosp_power_analysis_norm( float(prospective_mu),
                                                                 float(prospective_stddev),
                                                                 float(prospective_desired_power),
                                                                 float(prospective_sig_alpha),
                                                                 prospective_alternative)
        if debug: print("PROSPECTIVE_POWER: required sample size = {} for power = {}".format(
            prospective_required_sample, prospective_desired_power
        ))

        rendered = render_template(template_filename,
                                   prospective_mu = prospective_mu,
                                   prospective_sig_alpha = prospective_sig_alpha,
                                   prospective_stddev = prospective_stddev,
                                   prospective_alternative= prospective_alternative,
                                   prospective_desired_power=prospective_desired_power,
                                   prospective_required_sample=prospective_required_sample,
                                   last_tab_name_clicked='Prospective Power',
                                   # # data analysis tab
                                   # skewness_gamma=json.loads(request.cookies.get('skewness_gamma')),
                                   # normality_alpha=json.loads(request.cookies.get('normality_alpha')),
                                   # eval_unit_size=request.cookies.get('eval_unit_size'),
                                   # eval_unit_stat=request.cookies.get('eval_unit_stat'),
                                   # num_eval_units=request.cookies.get('num_eval_units'),
                                   # shuffle_seed=request.cookies.get('shuffle_seed'),
                                   # sigtest_heading=request.cookies.get('sig_test_heading'),
                                   # summary_str=request.cookies.get('summary_str'),
                                   # mean_or_median=request.cookies.get('mean_or_median'),
                                   # is_normal=json.loads(request.cookies.get('is_normal')),
                                   # recommended_tests=json.loads(request.cookies.get('recommended_tests')),
                                   # not_preferred_tests=json.loads(request.cookies.get('not_preferred_tests')),  # list of tuples
                                   # not_recommended_tests=json.loads(request.cookies.get('not_recommended_tests')),
                                   # show_non_recommended=request.form.get('checkbox_show_non_recommended'),
                                   # show_non_preferred=request.form.get('checkbox_show_non_preferred'),
                                   # summary_stats_list=json.loads(request.cookies.get('summary_stats_list')),
                                   # hist_score1_file=request.cookies.get('hist_score1_file'),
                                   # hist_score2_file=request.cookies.get('hist_score2_file'),
                                   # hist_diff_file=request.cookies.get('hist_diff_file'),
                                   # hist_diff_par_file=request.cookies.get('hist_diff_par_file'),
                                   # # specific to sig_test
                                   # mu=mu,
                                   # sig_test_stat_val=test_stat_val,
                                   # CI=CI,
                                   # pval=pval,
                                   # rejectH0=rejection,
                                   # sig_alpha=sig_alpha,
                                   # sig_test_name=sig_test_name,
                                   #wilcoxon_ci_rec=wilcoxon_ci_rec,
                                   #wilcoxon_ci_nonpref=wilcoxon_ci_nonpref,
                                   # specific to effect size test
                                   effect_size_estimators=estimators,
                                   eff_estimator=request.cookies.get('eff_estimator'),
                                   eff_size_val=request.cookies.get('eff_size_val'),
                                   rand_str=get_rand_state_str()
                                   )
        resp = make_response(rendered)
        # -------- WRITE TO COOKIES ----------
        # resp.set_cookie('sig_test_name', sig_test_name)
        # resp.set_cookie('sig_test_alpha', sig_alpha)
        # resp.set_cookie('alternative', alternative)
        # if test_stat_val:
        #     resp.set_cookie('sig_test_stat_val', json.dumps(float(test_stat_val)))
        #     print('test_stat_val={}, json_dumped={}'.format(test_stat_val, json.dumps(float(test_stat_val))))
        # if CI:
        #     resp.set_cookie('CI', json.dumps(CI))
        # if show_non_recommended:
        #     resp.set_cookie('show_non_recommended', show_non_recommended)
        # else:
        #     resp.set_cookie('show_non_recommended', '')
        # if show_non_preferred:
        #     resp.set_cookie('show_non_preferred', show_non_preferred)
        # else:
        #     resp.set_cookie('show_non_preferred', '')
        # if wilcoxon_ci_nonpref:
        #     resp.set_cookie('wilcoxon_ci_nonpref', wilcoxon_ci_nonpref)
        # else:
        #     resp.set_cookie('wilcoxon_ci_nonpref', '')
        # if wilcoxon_ci_nonpref:
        #     resp.set_cookie('wilcoxon_ci_rec', wilcoxon_ci_rec)
        # else:
        #     resp.set_cookie('wilcoxon_ci_rec', '')
        # resp.set_cookie('sig_boot_iterations', str(sig_boot_iterations))
        # resp.set_cookie('mu', str(mu))
        # resp.set_cookie('pval', str(pval))
        # resp.set_cookie('rejectH0', str(rejection))
        return resp
    # GET
    return render_template(template_filename,
                           rand_str=get_rand_state_str()
                           )
# ********************************************************************************************
#   SIGNIFICANCE TEST
# ********************************************************************************************
@app.route('/sig_test', methods=["GET", "POST"])
def sigtest(debug=True):
    if request.method == "POST":
        # ------- Get cookies
        fileName = request.cookies.get('fileName')
        # ------- Get form data
        show_non_recommended = request.form.get('checkbox_show_non_recommended')
        show_non_preferred = request.form.get('checkbox_show_non_preferred')
        sig_test_name = request.form.get('target_sig_test')
        sig_alpha = request.form.get('significance_level')
        mu = float(request.form.get('mu'))
        if not mu:
            mu = 0.0
        alternative = request.form.get('alternative')
        sig_boot_iterations = int(request.form.get('sig_boot_iterations'))
        wilcoxon_ci_rec = request.form.get('checkbox_wilcoxon_ci1')
        wilcoxon_ci_nonpref = request.form.get('checkbox_wilcoxon_ci2')
        if debug:
            print(' ********* Running /sig_test')
            print('Sig_test_name={}, sig_alpha={}'.format(sig_test_name, sig_alpha))

        # ------- Test if 'last_tab' was sent
        last_tab_name_clicked = 'Significance Test'  # request.form.get('last_tab_input')
        if debug: print("***** LAST TAB (from POST): {}".format(last_tab_name_clicked))

        scores1, scores2 = read_score_file(FOLDER + "/" + fileName)  # todo: different FOLDER for session/user
        # get old dif
        score_dif = calc_score_diff(scores1, scores2)

        # use Partition Score to get new dif
        partitions = partition_score_no_hist(scores1, scores2, score_dif,
                                json.loads(request.cookies.get('eval_unit_size')),
                                shuffled=False,randomSeed=0,method=request.cookies.get('mean_or_median'))
        score_dif = partitions[2]

        # Adjust conf_int for Wilcoxon case
        #conf_inf = True
        if sig_test_name == 'wilcoxon':
            if wilcoxon_ci_rec or wilcoxon_ci_nonpref:
                conf_inf = True
            else:
                conf_inf = False
        else:
            conf_inf = True
        test_stat_val, pval, CI, rejection = run_sig_test(sig_test_name,  # 't'
                                                      score_dif,
                                                      float(sig_alpha),  # 0.05,
                                                      B=sig_boot_iterations,
                                                      alternative=alternative,
                                                      conf_int=conf_inf,
                                                      mu=mu)
        if debug: print("test_stat_val={}, pval={},"
                        "alternative={}, mu={}, CI={}, rejection={}".format(
            test_stat_val, pval, alternative, mu, CI, rejection))

        recommended_tests = json.loads(request.cookies.get('recommended_tests'))
        not_preferred_tests = json.loads(request.cookies.get('not_preferred_tests'))
        summary_stats_list = json.loads(request.cookies.get('summary_stats_list'))

        rendered = render_template(template_filename,
                                   skewness_gamma=json.loads(request.cookies.get('skewness_gamma')),
                                   normality_alpha=json.loads(request.cookies.get('normality_alpha')),
                                   # specific to effect size test
                                   effect_size_estimators=estimators,
                                   eff_estimator=request.cookies.get('eff_estimator'),
                                   eff_size_val=request.cookies.get('eff_size_val'),
                                   # file_uploaded = "File uploaded!!: {}".format(fileName),
                                   last_tab_name_clicked=last_tab_name_clicked,
                                   # get from cookies
                                   eval_unit_size=request.cookies.get('eval_unit_size'),
                                   eval_unit_stat=request.cookies.get('eval_unit_stat'),
                                   num_eval_units=request.cookies.get('num_eval_units'),
                                   shuffle_seed=request.cookies.get('shuffle_seed'),
                                   sigtest_heading=request.cookies.get('sig_test_heading'),
                                   # todo: add teststat_heading
                                   summary_str=request.cookies.get('summary_str'),
                                   mean_or_median=request.cookies.get('mean_or_median'),
                                   is_normal=json.loads(request.cookies.get('is_normal')),
                                   recommended_tests=recommended_tests,
                                   not_preferred_tests=not_preferred_tests,  # list of tuples
                                   not_recommended_tests=json.loads(request.cookies.get('not_recommended_tests')),
                                   show_non_recommended=show_non_recommended,
                                   show_non_preferred=show_non_preferred,
                                   summary_stats_list=summary_stats_list,
                                   hist_score1_file=request.cookies.get('hist_score1_file'),
                                   hist_score2_file=request.cookies.get('hist_score2_file'),
                                   hist_diff_file=request.cookies.get('hist_diff_file'),
                                   hist_diff_par_file=request.cookies.get('hist_diff_par_file'),
                                   # specific to sig_test
                                   mu=mu,
                                   sig_boot_iterations=sig_boot_iterations,
                                   alternative=alternative,
                                   sig_test_stat_val=test_stat_val,
                                   CI=CI,
                                   pval=pval,
                                   rejectH0=rejection,
                                   sig_alpha=sig_alpha,
                                   sig_test_name=sig_test_name,
                                   wilcoxon_ci_rec=wilcoxon_ci_rec,
                                   wilcoxon_ci_nonpref=wilcoxon_ci_nonpref,
                                   rand_str=get_rand_state_str()
                                   )
        resp = make_response(rendered)
        # -------- WRITE TO COOKIES ----------
        resp.set_cookie('sig_test_name', sig_test_name)
        resp.set_cookie('sig_test_alpha', sig_alpha)
        resp.set_cookie('alternative', alternative)
        if test_stat_val:
            resp.set_cookie('sig_test_stat_val', json.dumps(float(test_stat_val)))
            print('test_stat_val={}, json_dumped={}'.format(test_stat_val, json.dumps(float(test_stat_val))))
        if CI:
            resp.set_cookie('CI', json.dumps(CI))
        if show_non_recommended:
            resp.set_cookie('show_non_recommended', show_non_recommended)
        else:
            resp.set_cookie('show_non_recommended', '')
        if show_non_preferred:
            resp.set_cookie('show_non_preferred', show_non_preferred)
        else:
            resp.set_cookie('show_non_preferred', '')
        if wilcoxon_ci_nonpref:
            resp.set_cookie('wilcoxon_ci_nonpref', wilcoxon_ci_nonpref)
        else:
            resp.set_cookie('wilcoxon_ci_nonpref', '')
        if wilcoxon_ci_nonpref:
            resp.set_cookie('wilcoxon_ci_rec', wilcoxon_ci_rec)
        else:
            resp.set_cookie('wilcoxon_ci_rec', '')
        resp.set_cookie('sig_boot_iterations', str(sig_boot_iterations))
        resp.set_cookie('mu', str(mu))
        resp.set_cookie('pval', str(pval))
        resp.set_cookie('rejectH0', str(rejection))
        return resp
    # GET
    return render_template(template_filename,
                           rand_str=get_rand_state_str()
                           )


@app.route('/effectsize', methods=["GET", "POST"])
def effectsize(debug=True):
    if request.method == 'POST':
        last_tab_name_clicked = 'Effect Size'
        fileName = request.cookies.get('fileName')
        scores1, scores2 = read_score_file(FOLDER + "/" + fileName)  # todo: different FOLDER for session/user
        # alpha for significance, boostrap iterations
        sig_test_alpha = json.loads(request.cookies.get('sig_test_alpha'))
        effectsize_sig_alpha = request.form.get('effectsize_significance_level')
        sig_boot_iterations = json.loads(request.cookies.get('sig_boot_iterations'))
        # get old dif
        score_dif = calc_score_diff(scores1, scores2)
        # use Partition Score to get new dif
        partitions = partition_score_no_hist(scores1, scores2, score_dif,
                                json.loads(request.cookies.get('eval_unit_size')),
                                shuffled=False,randomSeed=0,method=request.cookies.get('mean_or_median'))
        score_dif = partitions[2]
        previous_selected_est = request.cookies.get('eff_estimator')

        # todo: """                                        {% if key=='hl' %}Hodges-Lehmann Estimator
        #                                             {% elif key=='wilcoxonr' %}Wilcoxon r
        #                                             {% elif key=='hedgesg' %}Hedges' g
        #                                             {% elif key=='cohend' %}Cohen's d"""
        cur_selected_ests = []
        cur_selected_est_hl = request.form.get('target_eff_test_hl')
        if cur_selected_est_hl: cur_selected_ests.append(cur_selected_est_hl)
        cur_selected_est_wilcoxonr = request.form.get('target_eff_test_wilcoxonr')
        if cur_selected_est_wilcoxonr: cur_selected_ests.append(cur_selected_est_wilcoxonr)
        cur_selected_est_hedgesg = request.form.get('target_eff_test_hedgesg')
        if cur_selected_est_hedgesg: cur_selected_ests.append(cur_selected_est_hedgesg)
        cur_selected_est_cohend = request.form.get('target_eff_test_cohend')
        if cur_selected_est_cohend: cur_selected_ests.append(cur_selected_est_cohend)
        print('currentEstimators={}:'.format(cur_selected_ests))

        # old:
        # (estimates, estimators) = calc_eff_size(cur_selected_test,
        #                                         effect_size_target_stat,
        #                                         score_dif)


        # Build list of tuples for (estimator, value) pairs
        estimator_value_list = []
        for est in cur_selected_ests:
            val = calc_eff_size(est,
                                score_dif,
                                float(effectsize_sig_alpha),
                                #sig_test_alpha,
                                sig_boot_iterations) # sig_test_alpha, sig_boot_iterations
            estimator_value_list.append((est, val))

        # For completing previous tabs: target_stat is 'mean' or 'median'
        previous_selected_test = request.cookies.get('sig_test_name')
        recommended_tests = json.loads(request.cookies.get('recommended_tests'))
        summary_stats_list = json.loads(request.cookies.get('summary_stats_list'))
        print("EFFECT SIZE (from cookie): is_normal={}".format(json.loads(request.cookies.get('is_normal'))))
        skewness_gamma = json.loads(request.cookies.get('skewness_gamma'))
        rendered = render_template(template_filename,
                                   skewness_gamma=skewness_gamma,
                                   # specific to effect size test
                                   effect_size_estimators=estimators,  # just names
                                   estimator_value_list=estimator_value_list,  # name, value pairs
                                   effectsize_sig_alpha=effectsize_sig_alpha,
                                   # file_uploaded = "File uploaded!!: {}".format(fileName),
                                   last_tab_name_clicked=last_tab_name_clicked,
                                   # get from cookies
                                   eval_unit_size=request.cookies.get('eval_unit_size'),
                                   eval_unit_stat=request.cookies.get('eval_unit_stat'),
                                   num_eval_units=request.cookies.get('num_eval_units'),
                                   shuffle_seed=request.cookies.get('shuffle_seed'),
                                   sigtest_heading=request.cookies.get('sig_test_heading'),
                                   # todo: add teststat_heading
                                   summary_str=request.cookies.get('summary_str'),
                                   mean_or_median=request.cookies.get('mean_or_median'),
                                   is_normal=json.loads(request.cookies.get('is_normal')),
                                   recommended_tests=recommended_tests,
                                   not_preferred_tests=json.loads(request.cookies.get('not_preferred_tests')),
                                   not_recommended_tests=json.loads(request.cookies.get('not_recommended_tests')),
                                   show_non_preferred=request.cookies.get('show_non_preferred'),
                                   show_non_recommended=request.cookies.get('show_non_recommended'),
                                   summary_stats_list=summary_stats_list,
                                   hist_score1_file=request.cookies.get('hist_score1_file'),
                                   hist_score2_file=request.cookies.get('hist_score2_file'),
                                   hist_diff_file=request.cookies.get('hist_diff_file'),
                                   hist_diff_par_file=request.cookies.get('hist_diff_par_file'),
                                   # specific to sig_test
                                   sig_test_stat_val=request.cookies.get('sig_test_stat_val'),
                                   CI=request.cookies.get('CI'),
                                   pval=request.cookies.get('pval'),
                                   rejectH0=request.cookies.get('rejectH0'),
                                   sig_alpha=request.cookies.get('sig_test_alpha'),
                                   sig_test_name=request.cookies.get('sig_test_name'),
                                   alternative=request.cookies.get('alternative'),
                                   mu=request.cookies.get('mu'),
                                   wilcoxon_ci_rec=request.cookies.get('wilcoxon_ci_rec'),
                                   wilcoxon_ci_nonpref=request.cookies.get('wilcoxon_ci_nonpref'),
                                   rand_str=get_rand_state_str()
                                   )

        resp = make_response(rendered)
        # -------- WRITE TO COOKIES ----------
        resp.set_cookie('effect_estimator_dict', json.dumps(estimators))
        resp.set_cookie('estimator_value_list', json.dumps(estimator_value_list))
        resp.set_cookie('effectsize_sig_alpha', effectsize_sig_alpha)
        return resp

    elif request.method == 'GET':
        # You got to the main page by navigating to the URL, not by clicking submit
        # full_filename1 = os.path.join(app.config['FOLDER'], 'hist_score1.svg')
        # full_filename2 = os.path.join(app.config['FOLDER'], 'hist_score2.svg')
        return render_template('tab_interface.html',
                               rand_str=get_rand_state_str()
                               )


@app.route('/power', methods=["GET", "POST"])
def power(debug=True):
    if request.method == "POST":
        last_tab_name_clicked = 'Post-test Power Analysis'
        fileName = request.cookies.get('fileName')
        scores1, scores2 = read_score_file(FOLDER + "/" + fileName)  # todo: different FOLDER for session/user
        # get old dif
        score_dif = calc_score_diff(scores1, scores2)
        # use Partition Score to get new dif
        partitions = partition_score_no_hist(scores1, scores2, score_dif,
                                json.loads(request.cookies.get('eval_unit_size')),
                                shuffled=False,randomSeed=0,method=request.cookies.get('mean_or_median'))
        score_dif = partitions[2]
        is_normal = json.loads(request.cookies.get('is_normal'))
        print("POWER: (from cookie): is_normal={}".format(is_normal))

        if is_normal:
            power_test = request.form.get('target_pow_test')
        else:
            power_test = request.form.get('target_pow_test_bootstrap')

        power_num_intervals = int(request.form.get('num_intervals'))  # todo: get from form
        # if request.cookies.get('power_iterations'):
        power_iterations = int(request.form.get('power_iterations'))

        old_sig_test_name = request.cookies.get('sig_test_name')
        sig_test_name = request.form.get('power_boot_sig_test')
        if request.cookies.get('sig_test_alpha'):
            # actually, sigtest should be populating PA form alpha in template.
            # can do that instead of cookie.
            alpha = float(request.cookies.get('sig_test_alpha'))
        elif request.form.get('alpha'):
            alpha = float(request.form.get('alpha'))
        else:
            alpha = 0.05
        if request.form.get('mu'):
            mu = float(request.cookies.get('mu'))
        else:
            mu = 0
        if request.cookies.get('sig_boot_iterations'):
            boot_B = int(request.cookies.get('sig_boot_iterations'))
        else:
            boot_B = 500
        print('In PowerAnalysis: sig_test_name={} alpha={} mu={} bootB={} pow_iter={}'.format(
            sig_test_name, alpha, mu, boot_B, power_iterations))
        print("score dif: {}".format(score_dif))
        pow_sampsizes = post_power_analysis(sig_test_name, power_test, score_dif, power_num_intervals,
                                            dist_name='normal',  # todo: handle not normal
                                            B=power_iterations,
                                            alpha=alpha,
                                            mu=mu,
                                            boot_B=boot_B,
                                            output_dir=FOLDER)
        print(pow_sampsizes)

        power_file = 'power_samplesizes.svg'
        rand = np.random.randint(10000)
        # power_path = os.path.join(app.config['FOLDER'], power_file)

        recommended_tests = json.loads(request.cookies.get('recommended_tests'))
        summary_stats_list = json.loads(request.cookies.get('summary_stats_list'))

        skewness_gamma = json.loads(request.cookies.get('skewness_gamma'))
        rendered = render_template(template_filename,
                                   skewness_gamma=skewness_gamma,
                                   # power
                                   # old_sig_test_name=old_sig_test_name,
                                   power_file=power_file,
                                   power_test=power_test,
                                   power_num_intervals=power_num_intervals,
                                   power_iterations=power_iterations,
                                   # random number for forcing reload of images
                                   rand=rand,
                                   # specific to effect size test
                                   effect_size_estimators=estimators,
                                   eff_estimator=request.cookies.get('eff_estimator'),
                                   estimator_value_list=json.loads(request.cookies.get('estimator_value_list')),
                                   effectsize_sig_alpha=request.cookies.get('effectsize_sig_alpha'),
                                   eff_size_val=request.cookies.get('eff_size_val'),
                                   # effect_size_estimates = estimates,
                                   effect_estimator_dict = json.loads(request.cookies.get('effect_estimator_dict')),
                                   # file_uploaded = "File uploaded!!: {}".format(fileName),
                                   last_tab_name_clicked=last_tab_name_clicked,
                                   # get from cookies
                                   eval_unit_size=request.cookies.get('eval_unit_size'),
                                   eval_unit_stat=request.cookies.get('eval_unit_stat'),
                                   num_eval_units=request.cookies.get('num_eval_units'),
                                   shuffle_seed=request.cookies.get('shuffle_seed'),
                                   sigtest_heading=request.cookies.get('sig_test_heading'),
                                   # todo: add teststat_heading
                                   summary_str=request.cookies.get('summary_str'),
                                   mean_or_median=request.cookies.get('mean_or_median'),
                                   is_normal=json.loads(request.cookies.get('is_normal')),
                                   not_preferred_tests=json.loads(request.cookies.get('not_preferred_tests')),
                                   not_recommended_tests=json.loads(request.cookies.get('not_recommended_tests')),
                                   recommended_tests=recommended_tests,
                                   show_non_recommended=request.cookies.get('show_non_recommended'),
                                   show_non_preferred=request.cookies.get('show_non_preferred'),
                                   summary_stats_list=summary_stats_list,
                                   hist_score1_file=request.cookies.get('hist_score1_file'),
                                   hist_score2_file=request.cookies.get('hist_score2_file'),
                                   hist_diff_file=request.cookies.get('hist_diff_file'),
                                   hist_diff_par_file=request.cookies.get('hist_diff_par_file'),
                                   # specific to sig_test
                                   sig_test_stat_val=request.cookies.get('sig_test_stat_val'),  # json.loads?
                                   pval=request.cookies.get('pval'),
                                   CI=request.cookies.get('CI'),
                                   rejectH0=request.cookies.get('rejectH0'),
                                   sig_alpha=request.cookies.get('sig_test_alpha'),
                                   sig_test_name=sig_test_name,  # request.cookies.get('sig_test_name')
                                   rand_str=get_rand_state_str()
                                   )

        resp = make_response(rendered)
        # -------- WRITE TO COOKIES ----------
        # resp.set_cookie('pow_sampsizes', json.dumps(pow_sampsizes))
        resp.set_cookie('power_test', power_test)
        resp.set_cookie('sig_test_name', sig_test_name)
        resp.set_cookie('power_num_intervals', str(power_num_intervals))
        return resp
    return render_template(template_filename,
                           rand_str=get_rand_state_str()
                           )

@app.route('/upload_config', methods=["POST"])
def upload_config():
    print('Uploading from config file.')
    f = request.files['system_file']
    if f.filename:
        config_filename = f.filename
        f.save(FOLDER + "/" + secure_filename(config_filename))
        # Read YAML file
        with open(FOLDER + "/" + config_filename, 'r') as stream: # read_score_file(FOLDER + "/" + data_filename)
            data_loaded = yaml.safe_load(stream)
            print('Data loaded from config.yml: {}'.format(data_loaded))
            # TODO: get each field out of data_loaded to use in render_template parameters
            resp = render_template(template_filename,
                CI = data_loaded.get('CI'),
                alternative = data_loaded.get('alternative'),
                effect_estimator_dict = json.loads(data_loaded.get('effect_estimator_dict')),
                effectsize_sig_alpha = data_loaded.get('effectsize_sig_alpha'),
                estimator_value_list = json.loads(data_loaded.get('estimator_value_list')),
                eval_unit_size = data_loaded.get('eval_unit_size'),
                eval_unit_stat = data_loaded.get('eval_unit_stat'),
                fileName = data_loaded.get('fileName'),
                hist_diff_file = data_loaded.get('hist_diff_file'),
                hist_diff_par_file = data_loaded.get('hist_diff_par_file'),
                hist_score1_file = data_loaded.get(('hist_score1_file')),
                hist_score2_file = data_loaded.get(('hist_score2_file')),
                is_normal = data_loaded.get('is_normal'),
                mean_or_median = data_loaded.get('mean_or_median'),
                mu = data_loaded.get('mu'),
                normality_alpha= data_loaded.get('normality_alpha'),
                not_preferred_tests= json.loads(data_loaded.get('not_preferred_tests')),
                not_recommended_tests= json.loads(data_loaded.get('not_recommended_tests')),
                num_eval_units=  data_loaded.get('num_eval_units'),
                power_num_intervals= data_loaded.get('power_num_intervals'),
                power_test= data_loaded.get('power_test'),
                pval=data_loaded.get('pval'),
                recommended_tests = json.loads(data_loaded.get('recommended_tests')),
                rejectH0 = data_loaded.get('rejectH0'),
                show_non_preferred = data_loaded.get('show_non_preferred'),
                show_non_recommended = data_loaded.get(('show_non_recommended')),
                shuffle_seed= data_loaded.get('shuffle_seed'),
                sig_boot_iterations = json.loads(data_loaded.get('sig_boot_iterations')),
                sig_test_alpha = json.loads(data_loaded.get('sig_test_alpha')),
                sig_test_heading = data_loaded.get('sig_test_heading'),
                sig_test_name = data_loaded.get('sig_test_name'),
                sig_test_stat_val = data_loaded.get('sig_test_stat_val'),
                skewness_gamma = json.loads(data_loaded.get('skewness_gamma')),
                summary_stats_list = json.loads(data_loaded.get('summary_stats_list')),
                summary_str = data_loaded.get('summary_str'),
                rand_str=get_rand_state_str())
    else:
        # todo: handle exceptions, print or indicate that no file was chosen.
        resp = render_template(template_filename,
                               rand_str=get_rand_state_str())
    return resp

# https://www.roytuts.com/how-to-download-file-using-python-flask/
@app.route('/download')
def download_file(markdown_only=True):
    '''

    @param markdown_only: Set this to true to debug the case in which you don't want
    the whole zip file (like if zip isn't working)
    @return:
    '''

    options = {}
    options["filename"] = request.cookies.get('fileName')
    options["normality_message"] = request.cookies.get('is_normal')
    options["skewness_message"] = request.cookies.get('skewness_gamma')
    options["test_statistic_message"] = \
        request.cookies.get('mean_or_median')
    options["significance_tests_table"] = json.loads(request.cookies.get('recommended_tests'))
    options["significance_alpha"] = request.cookies.get('sig_test_alpha')
    options["bootstrap iterations"] = "200"
    # This was mu, which we're not letting the user define: options["expected_mean_diff"] = "0"
    options["chosen_sig_test"] = request.cookies.get('sig_test_name')
    options["should_reject?"] = request.cookies.get('rejectH0')
    options["statistic/CI"] = request.cookies.get('sig_test_stat_val')  #
    rand = np.random.randint(10000)
    gen_report(options, str(rand))
    if markdown_only:
        return send_file("user/report.md", as_attachment=True, cache_timeout=0)
    else:
        return send_file("user/report.zip", as_attachment=True, cache_timeout=0)


#@app.route('/download2')  # @app.route('/download2')
@app.route('/download2/<config_file_name>')
def download_config(config_file_name):   # was download_config() no param

    rand_str = str(np.random.randint(10000))
    config_file_path = 'user/'+config_file_name + rand_str + '.yml'
    # Note: this will also get cookies saved by other sites
    items = request.cookies.items()
    cookie_dict = {}
    for k,v in items:
        #print('key={}, value={}'.format(k,v))
        # todo: check if cookie in master list before adding
        cookie_dict[k]=v

    #Write YAML file
    print('Writing to file: {}'.format(config_file_path))
    with io.open(config_file_path, 'w', encoding='utf8') as outfile:
        yaml.dump(cookie_dict, outfile, default_flow_style=False, allow_unicode=True)

    return send_file(config_file_path, as_attachment=True, cache_timeout=0)
    #return send_file('user/config_cookies.yaml', as_attachment=True)

@app.route('/img_url/<image_path>')
def send_img_file(image_path, debug=False):
    '''
    if the file path is known to be a relative path then alternatively:
        return send_from_directory(dir_name, image_name)
    @param image_path: The full path to the image
    @param debug: print out the path
    @return:
    '''
    if debug: print('display image: {}'.format(image_path))
    return send_file(image_path, cache_timeout=0)


@app.route('/img_url_dir/<image_name>')
def send_img_file_dir(image_name, debug=False):
    '''
    if the file path is known to be a relative path then alternatively:
        return send_from_directory(dir_name, image_name)
    @param image_name: The filename of the image (don't include directory)
    @param debug: print out the path
    @return:
    '''
    dir_name = app.config['FOLDER']  # user
    if debug: print('display image: {}'.format(image_name))
    return send_from_directory(dir_name, image_name)


# --- Begin functions relating to help and manual ----
@app.route('/<help_file_name>', methods=["GET", "POST"])
def request_help_file(help_file_name='about.html', debug=True):
    # todo: reuse code from help functions
    if debug:
        print('Optional Filename parameter to request_help_file: {}'.format(help_file_name))
    try:
        file = send_file('./static/{}'.format(help_file_name), as_attachment=False, cache_timeout=0)
    except FileNotFoundError:
        if 'manual' in help_file_name or 'help' in help_file_name:
            file = send_file('./static/{}'.format('manual.html'), cache_timeout=0)
        elif 'about' in help_file_name:
            file = send_file('./static/{}'.format('about.html'), cache_timeout=0)
        else:
            # go to step 1 (todo: use previous state)
            file = render_template(template_filename, rand_str=get_rand_state_str())
    return file

@app.route('/manual')
def get_manual(debug=True):
    if debug: print('getting manual')
    try:
        # file = send_file('./static/manual.html', as_attachment=False, cache_timeout=0)
        file = send_file(get_path('manual_path'), as_attachment=False, cache_timeout=0)
    except FileNotFoundError:
        file = render_template(template_filename, rand_str=get_rand_state_str())
    return file

@app.route('/help/<help_file_name>/')
def get_help(help_file_name, debug=True):
    if debug: print('get_help: {}'.format(help_file_name))
    try:
        file = send_file('./static/{}'.format(help_file_name), as_attachment=False, cache_timeout=0)
    except:
        if 'manual' in help_file_name:
            file = send_file('./static/{}'.format('manual.html'), cache_timeout=0)
        elif 'about' in help_file_name:
            file = send_file('./static/{}'.format('about.html'), cache_timeout=0)
        else:
            file = render_template(template_filename, rand_str=get_rand_state_str())
    return file

if __name__ == "__main__":
    app.debug = True
    app.run()
