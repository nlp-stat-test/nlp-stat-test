
def get_path(path):
    manual_path = './static/manual.pdf'
    manual_filename = 'manual.pdf'
    path_dict = {
        'manual_path':'./static/manual.pdf',
        'manual_filename':'manual.pdf',
        'template_filename':'tab_interface.html'
    }
    return path_dict.get(path, '')