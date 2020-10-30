
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

def split_filename(filename, eu_size=1, seed=None):
   '''
    Used for naming the file with the repartitioned EU data
    @param filename: the original filename
    @param eu_size: size of the EU
    @param seed: seed, if provided
    @return: Filename
   '''
   x = filename.split(".")
   if len(x) >= 2:
        fname_start = '.'.join(x[:len(x)-1])
        fname_ext = x[-1]
   elif len(x) == 1:
        fname_start = x[0]
        fname_ext = 'txt'
   if seed:
     seed_str = '-seed'+str(seed)
   else:
     seed_str= ''
   fname = '{}-eusize{}{}.{}'.format(fname_start,eu_size,seed_str,fname_ext)
   return fname