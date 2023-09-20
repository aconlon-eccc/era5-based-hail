import os

def make_dir(path, nf=0):
    new_path = path
    if nf > 0:
        new_path = '{}.{}'.format(path, nf)
    try:
        os.mkdir(new_path)
        return new_path
    except FileExistsError:
        nf +=1
        return make_dir(path, nf=nf)