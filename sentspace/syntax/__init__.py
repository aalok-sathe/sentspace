import subprocess
import os
from sentspace.syntax.features import Feature, Tree, DLT, LeftCorner
from pathlib import Path
import pdb

os.environ['PERL_BADLANG'] = '0'


def path_decorator(func):
    """Decorator that changes to and from the directory containing scripts
        for running DLT and Left Corner metrics before and after a function
        call respectively.
    """

    def wrapped(*args, **kwargs):
        ''' function that changes the directory to an expected directory,
            executes original function with supplied args, and changes
            back to the same directory we started from
        '''
        previous_pwd = os.getcwd()
        target = Path(__file__) 
        os.chdir(str(target.parent / 'utils'))
        # DEBUG
        # pdb.set_trace()
        result = func(*args, **kwargs)
        os.chdir(previous_pwd)
        # DEBUG
        # pdb.set_trace()
        return result.decode('utf-8').strip()

    return wrapped


def get_features(text:str=None, dlt:bool=False, left_corner:bool=False):
    """executes the syntactic features pipeline

    Args:
        text (str, optional): a string containing one or more sentences
                              to be computed features of. [None].
        dlt (bool, optional): calculate DLT feature? [False].
        left_corner (bool, optional): calculate Left Corner feature? [False].

    Returns:
        sentspace.syntax.features.Feature: a Feature instance with appropriate attributes
    """
    features = Feature()
    if dlt or left_corner:
        features.tree = Tree(compute_trees(parse_input(text)))
    else:
        return None

    if dlt:
        features.dlt = DLT(compute_feature('dlt.sh', features.tree.raw))
    if left_corner:
        features.left_corner = LeftCorner(compute_feature('leftcorner.sh', features.tree.raw))
    return features


def parse_input(source):
    # TODO: collect input from various sources
    tokens = None
    if os.path.isdir(source):
        raw = open(source, 'r').read()
        tokens = tokenize(raw)
    if type(source) == str:
        tokens = tokenize(source)
    return tokens


def validate_input():
    # TODO: we just want a string
    pass


@path_decorator
def tokenize(raw):
    cmd = ['bash', 'tokenize.sh', raw]
    tokens = subprocess.check_output(cmd)
    return tokens


@path_decorator
def compute_trees(tokens):
    cmd = ['bash', 'parse_trees.sh', tokens]
    trees = subprocess.check_output(cmd)
    return trees


@path_decorator
def compute_feature(feature, trees):
    cmd = ['bash', feature, trees]
    out = subprocess.check_output(cmd)
    return out
