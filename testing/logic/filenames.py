
def get_path(path):
    '''
    @param path: corresponds to a variable name used in run.py, as parameter to render_template()
    @return:
    '''
    manual_path = './static/manual.pdf'
    manual_filename = 'manual.pdf'
    path_dict = {
        'manual_path':'./static/manual.pdf',
        'manual_filename':'manual.pdf',
        'template_filename':'tab_interface.html',
        'hist_score1_file': 'hist_score1_EUs.svg',
        'hist_score2_file': 'hist_score2_EUs.svg',
        'hist_diff_file': 'hist_diff_EUs.svg',
        'hist_diff_par_file': 'hist_score_diff_EUs.svg',
        'eu_size_std_dev_file': 'eu_size_std_dev.svg'
    }
    return path_dict.get(path, '')