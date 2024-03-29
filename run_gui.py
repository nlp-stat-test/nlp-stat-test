# v3
import io
import traceback
import shutil
import yaml
import zipfile
from flask import Flask
from flask import *
from flask import render_template
from flask import make_response
import logging
import datetime

from werkzeug.utils import secure_filename
import os
import sys
import numpy as np

# Business Logic
from src.logic.fileReader import read_score_file, print_eu
from src.logic.testCase import testCase
from src.logic.effectSize import calc_eff_size
from src.logic.dataAnalysis import partition_score, \
    skew_test, normality_test, recommend_test, choose_eu
from src.logic.sigTesting import run_sig_test
from src.logic.power_analysis_norm import prosp_power_analysis_norm

# filenames
from src.logic.filenames import get_path, split_filename
from src.logic.errorHandling import InputError


from src.logic.powerAnalysis import post_power_analysis
import src.logic.powerAnalysis

os.umask(0000)

FOLDER = os.path.join('user')
ERRORS = os.path.join('logs')
session_count = 0

app = Flask(__name__, static_folder=os.path.join("src", "static"),
            template_folder=os.path.join("src", "templates"))
app.config['FOLDER'] = FOLDER

print("\nGo to http://localhost:5000 in your browser.\n")



# defaults
DEFAULT_SEED = None
DEFAULT_EVAL_SIZE = 1

# template filename
# Note: "tab_inteface2.html" has histograms before recommendations
template_filename = "interface.html"

# strings to use in UI
summary_str = "Summary of Statistics"
teststat_heading = "Test Statistic Recommendation"
sig_test_heading = 'Recommended Significance Tests'
estimators = {"cohend": "This function calculates the Cohen's \(d\) effect size estimator.",
              "hedgesg": "This function takes the Cohen's \(d\) estimate as an input and calculates the Hedges's \(g\).",
              "hl": "This function estimates the Hodges-Lehmann estimator for the input score.",
              "wilcoxonr": "This function calculates the standardized \(z\)-score (\(r\)) for the Wilcoxon signed-rank test.",}

def handle_exception(dir_str='', debug=False):
    if debug: print('***** handle_exception(dir_str={})'.format(dir_str))
    exc_info = sys.exc_info()
    if not dir_str:
        folder_name = get_rand_state_str()
    else:
        folder_name = dir_str
    if not os.path.exists(ERRORS + "/" + folder_name):
        os.makedirs(ERRORS + "/" + folder_name)
    with open(ERRORS + "/" + 'last_error.txt', 'w') as tmp_err:
        traceback.print_exception(*exc_info, file=tmp_err)
        # TODO, DELETE error_temp if not debugging
    with open(ERRORS + "/" + folder_name + "/ErrorLog.txt", "a") as f_err:
        f_err.write('----------------------\n')
        str_err = str(exc_info[0]) + '\n' + str(exc_info[1])
        f_err.write('Exception info:\n{}\n'.format(str_err))
        print('------\n{}\n------'.format(str_err))
        traceback.print_exception(*exc_info, file=f_err)
    exception_message = "Exception occurred:" + str(sys.exc_info()[0]) + ". \n Please email dpm3@uw.edu and paste the exception message in the body of the email."
    # print_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
    return make_response(render_template(template_filename,
                                         exception_message=exception_message, ##str_err,
                                         rand_str=get_rand_state_str()))

def log_session_number(session_name):
    '''

    @param session_name: string that indicates the random 4-digit number assigned in get_rand_state_str() to the session,
    or if doing pre-test analysis just indicating "pre-test-power"
    @return:
    '''
    global session_count 
    if not os.path.exists(ERRORS):
        os.makedirs(ERRORS)
    with open(os.path.join(ERRORS, 'anonymous_log.csv'), 'r') as f:
        session_count = len(f.readlines()) + 1
    with open(os.path.join(ERRORS, 'anonymous_log.csv'), 'a+') as f:
        print(str(session_count) + "\t" + session_name + '\t' + datetime.datetime.now().strftime("%m-%d-%y\t%H:%M:%S"), file=f)


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
    #print(eval_unit_size)
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

def format_file_label(filename, action):
    '''
    get string formatted with italics for display in HTML label
    @param filename: name of file
    @param action: 'uploaded' or 'selected'
    @return: string formatted with italics for display in HTML label
    '''
    return 'File {}: {}.'.format(action, filename)

def create_summary_stats_list(tc, debug=False):
    if debug: print('Score 1: mean={}, med={}, sd={}, min={}, max={}'.format(tc.eda.summaryStat_score1.mu,
                                                                             tc.eda.summaryStat_score1.med,
                                                                             tc.eda.summaryStat_score1.sd,
                                                                             tc.eda.summaryStat_score1.min_val,
                                                                             tc.eda.summaryStat_score1.max_val))
    summary_dict = []
    # 'score1'
    summary_dict.append(('Column 1', [('Mean', format_digits(tc.eda.summaryStat_score1.mu)),
                                    ('Median', format_digits(tc.eda.summaryStat_score1.med)),
                                    ('Std. Dev.', format_digits(tc.eda.summaryStat_score1.sd)),
                                    ('Minimum', format_digits(tc.eda.summaryStat_score1.min_val)),
                                    ('Maximum', format_digits(tc.eda.summaryStat_score1.max_val))]))
    summary_dict.append(('Column 2', [('Mean', format_digits(tc.eda.summaryStat_score2.mu)),
                                     ('Median', format_digits(tc.eda.summaryStat_score2.med)),
                                    ('Std. Dev.', format_digits(tc.eda.summaryStat_score2.sd)),
                                    ('Minimum', format_digits(tc.eda.summaryStat_score2.min_val)),
                                    ('Maximum', format_digits(tc.eda.summaryStat_score2.max_val))]))
    summary_dict.append(('Difference', [('Mean', format_digits(tc.eda.summaryStat_score_diff_par.mu)),
                                        ('Median', format_digits(tc.eda.summaryStat_score_diff_par.med)),
                                        ('Std. Dev.', format_digits(tc.eda.summaryStat_score_diff_par.sd)),
                                        ('Minimum', format_digits(tc.eda.summaryStat_score_diff_par.min_val)),
                                        ('Maximum', format_digits(tc.eda.summaryStat_score_diff_par.max_val))]))
    return summary_dict

def json_loads_safe(arg):
    if arg is not None:
        return json.loads(arg)
    else:
        return None

@app.route('/start')
def start():
    try:
        return render_template('interface.html',
                               rand_str=get_rand_state_str())
    except:
        ret = handle_exception()
    return ret

@app.route('/upload', methods=["POST"])
def upload(debug=False):
    #print('In /upload')
    recommended_eu = 1
    try:
        if request.method == "POST":
            #print('In /upload POST')
            # ------- Data File ----------------
            f = request.files['data_file']
            last_tab_name_clicked = 'Upload Files'
            have_file = False
            have_filename = False
            have_data = False
            str_err = ''
            if f.filename:
                have_filename = True
                data_filename = f.filename
                dir_str = get_rand_state_str()

                # log the session number we're using for dir_str
                log_session_number(dir_str)

                # save to dir_str
                if not os.path.exists(FOLDER + "/" + dir_str):
                    os.makedirs(FOLDER + "/" + dir_str)
                f.save(FOLDER + "/" +dir_str + "/" + secure_filename(data_filename))
                #print('Check directory {} for saved file'.format(dir_str))
                have_file = True  # assume the above worked
            # elif request.cookies.get('fileName'):
            #     print('no f.filename, getting cookie')
            #     data_filename = request.cookies.get('fileName')
            #     have_filename = True
            else:
                # no filename, print error message
                print('no f.filename, no cookie')
                str_err = 'You must submit a file.'
                print('ERROR: submitted without filename! You must (re)submit!')
            if have_filename:
                if debug: print('have filename:{}'.format(data_filename))
                try:
                    # todo: raise InputException if scores1 or scores2 empty due to bad file
                    scores1, scores2 = read_score_file(FOLDER + "/" + dir_str + "/" + data_filename)
                    # if not scores1 or not scores2:
                    #     raise InputError('linewitherror','each line must be two values separated by whitespace')
                    if len(scores1) > 0 and len(scores2):
                        have_data = True
                except InputError as e:
                    print('Exception: InputError. Line={}. {}'.format(e.line_num, e.message))
                    str_err = 'Error in line {} of file {}. {}'.format(e.line_num, data_filename, e.message)
                except:
                    # Todo: Can we print sys.stderr message here?
                    print('Exception occurred reading file: filename={}'.format(data_filename))
                    str_err = 'Exception occurred reading file: filename={}'.format(data_filename)
            parsed_config = False
            if have_data:
                # ------------- Config file (save only if data file was provided -----------
                config = request.files['config_file']
                if config.filename:
                    print('saving config file: {}'.format(config.filename))
                    config.save(FOLDER + "/" + dir_str + "/" + secure_filename(config.filename))
                    # Read YAML file
                    with open(FOLDER + "/" + dir_str + "/" + config.filename,
                              'r') as stream:  # read_score_file(FOLDER + "/" + data_filename)
                        data_loaded = yaml.safe_load(stream)
                        print('Data loaded from config.yml: {}'.format(data_loaded))
                        # check type of data_loaded (should be 'dict'. string will have no .get method)
                        if isinstance(data_loaded, dict):
                            parsed_config = True

                            rendered = render_template(template_filename,
                                               last_tab_name_clicked=last_tab_name_clicked,
                                               file_label=format_file_label(f.filename, 'uploaded'),
                                               config_file_label=format_file_label(config.filename, 'uploaded'),
                                               CI=data_loaded.get('CI'),
                                               alternative=data_loaded.get('alternative'),
                                               effect_estimator_dict=json_loads_safe(data_loaded.get('effect_estimator_dict')),
                                               effectsize_sig_alpha=data_loaded.get('effectsize_sig_alpha'),
                                               estimator_value_list=json_loads_safe(data_loaded.get('estimator_value_list')),
                                               eval_unit_size=data_loaded.get('eval_unit_size'),
                                               eval_unit_stat=data_loaded.get('eval_unit_stat'),
                                               fileName=data_loaded.get('fileName'),
                                               eu_size_std_dev_file=data_loaded.get('eu_size_std_dev_file'),
                                               hist_diff_file=data_loaded.get('hist_diff_file'),
                                               hist_diff_par_file=data_loaded.get('hist_diff_par_file'),
                                               hist_score1_file=data_loaded.get(('hist_score1_file')),
                                               hist_score2_file=data_loaded.get(('hist_score2_file')),
                                               is_normal=data_loaded.get('is_normal'),
                                               mean_or_median=data_loaded.get('mean_or_median'),
                                               mu=data_loaded.get('mu'),
                                               normality_alpha=data_loaded.get('normality_alpha'),
                                               not_preferred_tests=json_loads_safe(data_loaded.get('not_preferred_tests')),
                                               not_recommended_tests=json_loads_safe(data_loaded.get('not_recommended_tests')),
                                               num_eval_units=json.loads(data_loaded.get('num_eval_units')),
                                               power_num_intervals=data_loaded.get('power_num_intervals'),
                                               power_test=data_loaded.get('power_test'),
                                               pval=data_loaded.get('pval'),
                                               recommended_tests=json_loads_safe(data_loaded.get('recommended_tests')),
                                               rejectH0=data_loaded.get('rejectH0'),
                                               show_non_preferred=data_loaded.get('show_non_preferred'),
                                               show_non_recommended=data_loaded.get(('show_non_recommended')),
                                               shuffle_seed=data_loaded.get('shuffle_seed'),
                                               sig_boot_iterations=json_loads_safe(data_loaded.get('sig_boot_iterations')),
                                               sig_test_alpha=json_loads_safe(data_loaded.get('sig_test_alpha')),
                                               sig_test_heading=data_loaded.get('sig_test_heading'),
                                               sig_test_name=data_loaded.get('sig_test_name'),
                                               sig_test_stat_val=data_loaded.get('sig_test_stat_val'),
                                               skewness_gamma=json_loads_safe(data_loaded.get('skewness_gamma')),
                                               summary_stats_list=json_loads_safe(data_loaded.get('summary_stats_list')),
                                               summary_str=data_loaded.get('summary_str'),
                                               rand_str=get_rand_state_str())
                        else: # the .yml wasn't parsed as a dict
                            config_str_err = 'Unable to parse config file. Config file ignored.'
                            rendered = render_template(template_filename,
                                                       fileName=data_filename,
                                                       last_tab_name_clicked=last_tab_name_clicked,
                                                       rand_str=get_rand_state_str(),
                                                       config_error_str=config_str_err,
                                                       file_label=f.filename
                                                       )
                else:  # no config file
                    scores1, scores2 = read_score_file(FOLDER + "/" + dir_str + "/" + data_filename)


                    eval_unit_stat = request.form.get('target_statistic')

                    seed = request.form.get('seed')
                    if not seed:
                         shuffle = False
                    else:
                         shuffle = True

                    recommended_eu, explanation, table = choose_eu(calc_score_diff(scores1, scores2), shuffle, seed, eval_unit_stat, "user/" + dir_str)
                    eu_size_std_dev_file = get_path('eu_size_std_dev_file'),
                    rendered = render_template(template_filename,
                                               last_tab_name_clicked=last_tab_name_clicked,
                                               rand_str=get_rand_state_str(),
                                               eu_size_std_dev_file = "img_url_dir/" + eu_size_std_dev_file[0],
                                               error_str=str_err,
                                               fileName = data_filename,
                                               file_label=format_file_label(f.filename, 'uploaded'),
                                               eu_table = explanation,
                                               eval_unit_size = recommended_eu,
                                       )
            else: # don't have data
                str_err = 'Check that your file is properly formatted before upload'
                file_err = 'was not properly formatted.'
                rendered = render_template(template_filename,
                                               rand_str=get_rand_state_str(),
                                               error_str=str_err,
                                               file_label=format_file_label(f.filename, file_err)
                                       )
            resp = make_response(rendered)

            # Set cookies
            if have_data:
                if f.filename:

                    resp.set_cookie('fileName', f.filename)
                    resp.set_cookie('file_label', "File selected: {}".format(f.filename))
                    if config.filename and parsed_config:
                        resp.set_cookie('config_file_label', format_file_label(config.filename, 'uploaded'))
                    if 'dir_str_list' in request.cookies:
                        old = list(json.loads(request.cookies.get('dir_str_list')))
                        print(json.loads(request.cookies.get('dir_str_list')))
                        resp.set_cookie('dir_str_list', json.dumps([dir_str] + old))
                    else:
                        resp.set_cookie('dir_str_list', json.dumps([dir_str]))
                    resp.set_cookie('dir_str', dir_str)
            return resp
    except:
        ret = handle_exception()
        return ret

@app.route('/', methods=["GET", "POST"])
def home():
    return redirect(url_for('landing_page'))

@app.route('/home', methods=["GET", "POST"])
def landing_page():
    #print('.... Landing page')

    return render_template('welcome.html')
    #return redirect(url_for('data_analysis'))


@app.route('/ppa', methods=["GET", "POST"])
def ppa():
    return render_template('ppa.html')


@app.route('/data_analysis', methods=["GET", "POST"])
def data_analysis(debug=False):
    str_err = ''
    try:
        if request.method == 'POST':
            # ------- Test if 'last_tab' was sent
            last_tab_clicked = request.form.get('last_tab') #todo: remove this line?
            # todo: make this a cookie
            last_tab_name_clicked = 'Data Analysis'  # request.form.get('last_tab_name')
            #print("***** LAST TAB: {}".format(last_tab_name_clicked))

            eval_unit_size = request.form.get("eval_unit_size")

            # Handle case of no eval unit size
            if not eval_unit_size:
                eval_unit_size = DEFAULT_EVAL_SIZE

            # target_stat is 'mean' or 'median'
            eval_unit_stat = request.form.get('target_statistic')
            #print('eval_unit_stat={}'.format(eval_unit_stat))

            # normality
            normality_alpha = float(request.form.get('normality_alpha'))
            #print('NORMALITY_ALPHA (from form)={}'.format(normality_alpha))
            seed = request.form.get('seed')
            if not seed:
                shuffle = False
                # seed = DEFAULT_SEED
            else:
                shuffle = True

            # Epsilon
            epsilon = float(request.form.get('epsilon'))
            #print('EPSILON (from form)={}'.format(epsilon))


            # ------- File ----------------
            have_file = False
            # f = request.files['data_file']  # old use of file input

            have_filename = False
            have_data = False
            # if f.filename:
            #     have_filename = True
            #     data_filename = f.filename
            #     f.save(FOLDER + "/" + secure_filename(data_filename))
            #     have_file = True  # assume the above worked
            if request.cookies.get('fileName'):
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
                    # scores1, scores2 = read_score_file(FOLDER + "/" + data_filename)
                    dir_str = request.cookies.get('dir_str')
                    if dir_str:
                         scores1, scores2 = read_score_file(FOLDER + "/" + dir_str + "/" + data_filename)
                         print ('Just read scores from directory {}'.format(dir_str))

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
                dir_folder = FOLDER + "/" + dir_str
                if debug: print('FOLDER/dir_str={}'.format(dir_folder))
                score_dif_par = partition_score(scores1, scores2, score_dif, float(eval_unit_size),
                                                 shuffle,  # shuffle if we have seed
                                                 seed,
                                                 eval_unit_stat,  # mean or median
                                                 dir_folder)
                new_score1 = score_dif_par[0]
                new_score2 = score_dif_par[1]
                new_score_dif_par = score_dif_par[2]
                ind_shuffled = score_dif_par[3]
                # --------------Summary Stats -------------
                ### initialize a new testCase object to use for summary statistics
                tc = testCase(new_score1, #score_diff_par[0],
                              new_score2, #score_diff_par[1],
                              score_dif,  # original difference between scores, before applying EUs
                              new_score_dif_par,  # score_diff_par[2],
                              num_eval_units)

                tc.get_summary_stats()

                summary_stats_list = create_summary_stats_list(tc)

                # -----------Save EU file -----------------
                # Todo: create a make_eu_score_filename() function that takes orig_filename, eusize, seed
                print_eu(new_score1, new_score2, new_score_dif_par, ind_shuffled, dir_folder,
                         filename=split_filename(data_filename, eval_unit_size, seed))

                # --------------Recommended Test Statistic (mean or median, by skewness test) ------------------
                mean_or_median = skew_test(new_score_dif_par)[1]
                skewness_gamma = skew_test(new_score_dif_par)[0]

                # ---------------normality test
                is_normal = normality_test(new_score_dif_par, alpha=normality_alpha)
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
                                   hist_score1_file='hist_score1_EUs.svg',
                                   hist_score2_file='hist_score2_EUs.svg')
                else:
                    rand = np.random.randint(10000)
                    if debug: print('random number to append to image url={}'.format(rand))

                    rendered = render_template(template_filename,
                                               file_label = request.cookies.get('file_label'),
                                               config_file_label = request.cookies.get('config_file_label'),
                                               normality_alpha=normality_alpha,
                                               skewness_gamma=skewness_gamma,

                                               hist_score1_file= get_path('hist_score1_file'),#'hist_score1_EUs.svg',
                                               hist_score2_file=get_path('hist_score2_file'),
                                               hist_diff_file=get_path('hist_diff_file'),
                                               hist_diff_par_file=get_path('hist_diff_par_file'),
                                               last_tab_name_clicked=last_tab_name_clicked,
                                               eval_unit_size=eval_unit_size,
                                               eval_unit_stat=eval_unit_stat,
                                               num_eval_units=num_eval_units,
                                               max_pow_int=num_eval_units/15, #todo
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
                                               rand_str=get_rand_state_str(),
                                               #EU_table= "\n".join(["<tr><td>" + str(x) + "</td><td>" + str(y) + "</td></tr>" for x,y in zip(*EU_table)])

                                               )
                    resp = make_response(rendered)

                    # -------------- Set all cookies -------------
                    resp.set_cookie('num_eval_units', json.dumps(num_eval_units)),
                    resp.set_cookie('last_tab', last_tab_name_clicked)
                    # if f.filename:
                    #     resp.set_cookie('fileName', f.filename)
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
                    #print('DA Set cookie: is_normal dumped={}'.format(json.dumps(is_normal)))
                    resp.set_cookie('is_normal', json.dumps(is_normal))

                    resp.set_cookie('sig_test_heading', sig_test_heading)
                    resp.set_cookie('recommended_tests', json.dumps(recommended_tests))
                    resp.set_cookie('not_recommended_tests', json.dumps(not_recommended_tests))
                    resp.set_cookie('not_preferred_tests', json.dumps(not_preferred_tests))

                    resp.set_cookie('hist_score1_file', get_path('hist_score1_file'))
                    resp.set_cookie('hist_score2_file', get_path('hist_score2_file'))
                    resp.set_cookie('hist_diff_file', get_path('hist_diff_file'))
                    resp.set_cookie('hist_diff_par_file', get_path('hist_diff_par_file'))
                    resp.set_cookie('eu_size_std_dev_file', get_path('eu_size_std_dev_file'))
                    return resp  # return rendered
            else:
                # no file
                rendered = render_template(template_filename,
                                           rand_str=get_rand_state_str(),
                                           error_str=str_err, )
                return rendered
        elif request.method == 'GET':
            # You got to the main page by navigating to the URL, not by clicking submit
            # TODO: get rid of 'helper'
            return render_template(template_filename,
                                   file_label="Upload a file.",
                                   recommended_tests=[],
                                   summary_stats_list={},
                                   rand_str=get_rand_state_str()
                                   )
    except:
        ret = handle_exception()
        return ret

# ************************************************************
#   PROSPECTIVE POWER
# ************************************************************
@app.route('/ppa_results', methods=["GET", "POST"])
def ppa_results(debug=False):
    try:
        if request.method == "POST":
            log_session_number("prospective-power")
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

            rendered = render_template("ppa.html",
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
        return render_template("ppa.html",
                               rand_str=get_rand_state_str()
                               )
    except:
        ret = handle_exception()
        return ret
# ********************************************************************************************
#   SIGNIFICANCE TEST
# ********************************************************************************************
@app.route('/sig_test', methods=["GET", "POST"])
def sigtest(debug=False):
    try:
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

            dir_str = request.cookies.get('dir_str')
            scores1, scores2 = read_score_file(FOLDER + "/" + dir_str + "/"+ fileName)

            # get old dif
            score_dif = calc_score_diff(scores1, scores2)

            # use Partition Score to get new dif
            partitions = partition_score_no_hist(scores1, scores2, score_dif,
                                    json.loads(request.cookies.get('eval_unit_size')),
                                    shuffled=False, randomSeed=0,method=request.cookies.get('mean_or_median'))
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
                                       file_label=request.cookies.get('file_label'),
                                       config_file_label=request.cookies.get('config_file_label'),
                                       skewness_gamma=json.loads(request.cookies.get('skewness_gamma')),
                                       normality_alpha=json.loads(request.cookies.get('normality_alpha')),
                                       # specific to effect size test
                                       effect_size_estimators=estimators,
                                       eff_estimator=request.cookies.get('eff_estimator'),
                                       eff_size_val=request.cookies.get('eff_size_val'),
                                       last_tab_name_clicked=last_tab_name_clicked,
                                       # get from cookies
                                       eval_unit_size=request.cookies.get('eval_unit_size'),
                                       eval_unit_stat=request.cookies.get('eval_unit_stat'),
                                       num_eval_units=json.loads(request.cookies.get('num_eval_units')),
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
                                       eu_size_std_dev_file=request.cookies.get('eu_size_std_dev_file'),
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
            resp.set_cookie('last_tab', last_tab_name_clicked)
            resp.set_cookie('sig_test_name', sig_test_name)
            resp.set_cookie('sig_test_alpha', sig_alpha)
            resp.set_cookie('alternative', alternative)
            if test_stat_val:
                resp.set_cookie('sig_test_stat_val', json.dumps(float(test_stat_val)))
                #print('test_stat_val={}, json_dumped={}'.format(test_stat_val, json.dumps(float(test_stat_val))))
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
            if wilcoxon_ci_rec:
                resp.set_cookie('wilcoxon_ci_rec', wilcoxon_ci_rec)
            else:
                resp.set_cookie('wilcoxon_ci_rec', '')
            resp.set_cookie('sig_boot_iterations', str(sig_boot_iterations))
            resp.set_cookie('mu', str(mu))
            resp.set_cookie('pval', str(pval))
            resp.set_cookie('rejectH0', json.dumps(bool(rejection)))
            return resp
        # GET
        return render_template(template_filename,
                               rand_str=get_rand_state_str()
                               )
    except:
        ret = handle_exception()
        return ret


@app.route('/effectsize', methods=["GET", "POST"])
def effectsize(debug=False):
    try:
        if request.method == 'POST':
            last_tab_name_clicked = 'Effect Size'
            fileName = request.cookies.get('fileName')
            dir_str = request.cookies.get('dir_str')
            scores1, scores2 = read_score_file(FOLDER + "/" + dir_str + "/"+ fileName)
            num_eval_units = request.cookies.get('num_eval_units')
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
            cur_selected_est_wilcoxonr = request.form.get('target_eff_test_wilcoxonr')
            if cur_selected_est_wilcoxonr: cur_selected_ests.append(cur_selected_est_wilcoxonr)
            cur_selected_est_hl = request.form.get('target_eff_test_hl')
            if cur_selected_est_hl: cur_selected_ests.append(cur_selected_est_hl)
            cur_selected_est_hedgesg = request.form.get('target_eff_test_hedgesg')
            if cur_selected_est_hedgesg: cur_selected_ests.append(cur_selected_est_hedgesg)
            cur_selected_est_cohend = request.form.get('target_eff_test_cohend')
            if cur_selected_est_cohend: cur_selected_ests.append(cur_selected_est_cohend)
            #print('currentEstimators={}:'.format(cur_selected_ests.reverse()))

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
            #print("EFFECT SIZE (from cookie): is_normal={}".format(json.loads(request.cookies.get('is_normal'))))
            skewness_gamma = json.loads(request.cookies.get('skewness_gamma'))
            rendered = render_template(template_filename,
                                       file_label=request.cookies.get('file_label'),
                                       config_file_label=request.cookies.get('config_file_label'),
                                       skewness_gamma=skewness_gamma,
                                       # specific to effect size test
                                       effect_size_estimators=estimators,  # just names
                                       estimator_value_list=estimator_value_list,  # name, value pairs
                                       effectsize_sig_alpha=effectsize_sig_alpha,
                                       last_tab_name_clicked=last_tab_name_clicked,
                                       # get from cookies
                                       eval_unit_size=request.cookies.get('eval_unit_size'),
                                       eval_unit_stat=request.cookies.get('eval_unit_stat'),
                                       num_eval_units=json.loads(request.cookies.get('num_eval_units')),
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
                                       eu_size_std_dev_file=request.cookies.get('eu_size_std_dev_file'),
                                       # specific to sig_test
                                       sig_test_stat_val=request.cookies.get('sig_test_stat_val'),
                                       CI=request.cookies.get('CI'),
                                       pval=request.cookies.get('pval'),
                                       rejectH0=json.loads(request.cookies.get('rejectH0')),
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
            resp.set_cookie('last_tab', last_tab_name_clicked)
            resp.set_cookie('effect_estimator_dict', json.dumps(estimators))
            resp.set_cookie('estimator_value_list', json.dumps(estimator_value_list))
            resp.set_cookie('effectsize_sig_alpha', effectsize_sig_alpha)
            return resp

        elif request.method == 'GET':
            # You got to the main page by navigating to the URL, not by clicking submit
            # full_filename1 = os.path.join(app.config['FOLDER'], 'hist_score1.svg')
            # full_filename2 = os.path.join(app.config['FOLDER'], 'hist_score2.svg')
            return render_template('interface.html',
                                   rand_str=get_rand_state_str()
                                   )
    except:
        ret = handle_exception()
        return ret


@app.route('/power', methods=["GET", "POST"])
def power(debug=False):
    try:
        if request.method == "POST":
            num_eval_units = request.cookies.get('num_eval_units')
            last_tab_name_clicked = 'Retrospective Power Analysis'
            fileName = request.cookies.get('fileName')
            dir_str = request.cookies.get('dir_str')
            scores1, scores2 = read_score_file(FOLDER + "/" + dir_str + "/"+ fileName)
            # get old dif
            score_dif = calc_score_diff(scores1, scores2)
            # use Partition Score to get new dif
            partitions = partition_score_no_hist(scores1, scores2, score_dif,
                                    json.loads(request.cookies.get('eval_unit_size')),
                                    shuffled=False,randomSeed=0,method=request.cookies.get('mean_or_median'))
            score_dif = partitions[2]
            is_normal = json.loads(request.cookies.get('is_normal'))
            #print("POWER: (from cookie): is_normal={}".format(is_normal))

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
          # http://127.0.0.1:5000/ #print('In PowerAnalysis: sig_test_name={} alpha={} mu={} bootB={} pow_iter={}'.format(
               # sig_test_name, alpha, mu, boot_B, power_iterations))
            num_scores = len(score_dif)

            if (num_scores/power_num_intervals < 2):
                power_num_intervals = num_scores // 2
                #print("len(score)=={} \npower_num_intervals changed to {}".format(num_scores, power_num_intervals))
                # todo: create warning message to display in interface

            dir_str = request.cookies.get("dir_str")
            pow_sampsizes = post_power_analysis(sig_test_name, power_test, score_dif, power_num_intervals,
                                                dist_name='normal',  # todo: handle not normal
                                                B=power_iterations,
                                                alpha=alpha,
                                                mu=mu,
                                                boot_B=boot_B,
                                                output_dir=FOLDER + "/" + dir_str)
            #print(pow_sampsizes)

            power_file = 'power_samplesizes.svg'
            rand = np.random.randint(10000)
            # power_path = os.path.join(app.config['FOLDER'], power_file)

            recommended_tests = json.loads(request.cookies.get('recommended_tests'))
            summary_stats_list = json.loads(request.cookies.get('summary_stats_list'))

            skewness_gamma = json.loads(request.cookies.get('skewness_gamma'))
            rendered = render_template(template_filename,
                                       file_label=request.cookies.get('file_label'),
                                       config_file_label=request.cookies.get('config_file_label'),
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
                                       last_tab_name_clicked=last_tab_name_clicked,
                                       # get from cookies
                                       eval_unit_size=request.cookies.get('eval_unit_size'),
                                       eval_unit_stat=request.cookies.get('eval_unit_stat'),
                                       num_eval_units=json.loads(request.cookies.get('num_eval_units')),
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
                                       eu_size_std_dev_file=request.cookies.get('eu_size_std_dev_file'),
                                       # specific to sig_test
                                       sig_test_stat_val=request.cookies.get('sig_test_stat_val'),  # json.loads?
                                       alternative=request.cookies.get('alternative'),
                                       pval=request.cookies.get('pval'),
                                       CI=request.cookies.get('CI'),
                                       rejectH0=json.loads(request.cookies.get('rejectH0')),
                                       sig_alpha=request.cookies.get('sig_test_alpha'),
                                       sig_test_name=sig_test_name,  # request.cookies.get('sig_test_name')
                                       rand_str=get_rand_state_str()
                                       )

            resp = make_response(rendered)
            # -------- WRITE TO COOKIES ----------
            # resp.set_cookie('pow_sampsizes', json.dumps(pow_sampsizes))
            resp.set_cookie('last_tab', last_tab_name_clicked)
            resp.set_cookie('power_test', power_test)
            resp.set_cookie('sig_test_name', sig_test_name)
            resp.set_cookie('power_num_intervals', str(power_num_intervals))
            return resp
        # GET
        return render_template(template_filename,
                               rand_str=get_rand_state_str()
                               )
    except:
        ret = handle_exception()
        return ret

@app.route('/upload_config', methods=["POST", "GET"])
def upload_config():
    '''
    TODO:
       * add last_tab_name_clicked parameter.
       * make sure last_tab is previously saved in cookies in other steps
    @return:
    '''
    try:
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
                    eu_size_std_dev_file = data_loaded.get('eu_size_std_dev_file'),
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
    except:
        ret = handle_exception()
        return ret




@app.route('/download_config/<config_file_name>')
def download_config(config_file_name):   # was download_config() no param
    try:
        dir_str = request.cookies.get('dir_str')
        config_file_path = 'user/'+ dir_str + '/' + config_file_name + '.yml'
        # Note: this may also get cookies saved by other sites
        items = request.cookies.items()
        cookie_dict = {}
        for k,v in items:
            #print('key={}, value={}'.format(k,v))
            # todo: check if cookie in master list before adding
            cookie_dict[k]=v

        #Write YAML file
        #print('Writing to file: {}'.format(config_file_path))
        with io.open(config_file_path, 'w', encoding='utf8') as outfile:
            yaml.dump(cookie_dict, outfile, default_flow_style=False, allow_unicode=True)

        return send_file(config_file_path, as_attachment=True, cache_timeout=0)
        #return send_file('user/config_cookies.yaml', as_attachment=True)
    except:
        ret = handle_exception()
        return ret

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
    try:
        dir_str = request.cookies.get('dir_str')
        dir_name = app.config['FOLDER']  + '/' + str(dir_str) # user
        if debug: print('display image: {}/{}'.format(dir_name, image_name))
        return send_from_directory(dir_name, image_name)
    except:
        ret = handle_exception()
        return ret


@app.route('/manual')
def manual():
    try:
        file = send_file(os.path.join("src", "static", "manual.pdf"), as_attachment=False, cache_timeout=0)
        return file
    except:
        ret = handle_exception()
        return ret


@app.route('/paper')
def paper():
    try:
        file = send_file(os.path.join("src", "static", "paper.pdf"), as_attachment=False, cache_timeout=0)
    except FileNotFoundError:
        file = render_template(template_filename, rand_str=get_rand_state_str())
    return file


@app.route('/help/<help_file_name>/')
def get_help(help_file_name, debug=False):
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

def print_exception(cls, ex, traceback):
    print('{}\nException trace last instruction: {}'.format(ex, traceback.tb_lasti))
    return





# https://www.roytuts.com/how-to-download-file-using-python-flask/
@app.route('/download_zip')
def download_zip():
    try:
        zip_file = FOLDER + "/" + str(request.cookies.get("dir_str"))
        with zipfile.ZipFile(zip_file + ".zip",'w') as zip:
          for root, dirs, files in os.walk(zip_file):
            for name in files:
              zip.write(os.path.join(root, name), name)
        return send_file(zip_file +".zip", as_attachment=True, cache_timeout=0)
    except:
        ret = handle_exception()
        return ret

# https://www.roytuts.com/how-to-download-file-using-python-flask/
@app.route('/delete')
def delete_data():
    if "dir_str_list" in request.cookies:
      for dir_str in list(json.loads(request.cookies.get('dir_str_list'))):
          zip_file = FOLDER + "/" + dir_str
          if os.path.exists(zip_file):
              shutil.rmtree(zip_file)
          if os.path.isfile(zip_file + ".zip"):
              os.remove(zip_file + ".zip")
      print("Deleted user sessions: " + str(list(json.loads(request.cookies.get('dir_str_list')))))
    # https://stackoverflow.com/questions/14386304/flask-how-to-remove-cookies
    rendered = render_template("deleted.html")
    resp = make_response(rendered)
    resp.set_cookie("dir_str_list", json.dumps([]))
    resp.set_cookie("dir_str", '')
    return resp



if __name__ == "__main__":
    app.debug = False
    # https://stackoverflow.com/questions/14888799/disable-console-messages-in-flask-server
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log.disabled = True
    app.run()    # TODO: Allow argument
