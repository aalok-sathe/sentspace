import os
import subprocess
from functools import lru_cache
from pathlib import Path

from joblib import Memory
from nltk.tree import ParentedTree
from sentspace.syntax.features import DLT, Feature, LeftCorner, Tree

from sentspace.utils.caching import cache_to_disk
from sentspace.utils import io

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


def get_features(text:str=None, dlt:bool=False, left_corner:bool=False, parse_beam_width=5000):
    """executes the syntactic features pipeline

    Args:
        text (str, optional): paragraph of 1 or more sentences separated by newline or path to file [None].
        dlt (bool, optional): calculate DLT feature? [False].
        left_corner (bool, optional): calculate Left Corner feature? [False].

    Returns:
        sentspace.syntax.features.Feature: a Feature instance with appropriate attributes
    """
    features = Feature()
    if dlt or left_corner:
        io.log(f'parsing given text')
        parsed = parse_input(text)
        io.log(f'computing tree with beam_width={parse_beam_width} for parsed text')
        features.tree = Tree(compute_trees(parsed, beam_width=parse_beam_width))
        io.log(f'--- done: tree computed')
    else:
        return None

    # print(parse_input(text), features.tree)
    if dlt:
        io.log(f'computing DLT feature')
        features.dlt = DLT(compute_feature('dlt.sh', features.tree.raw))
        io.log(f'--- done: DLT')
    if left_corner:
        io.log(f'computing left corner feature')
        features.left_corner = LeftCorner(compute_feature('leftcorner.sh', features.tree.raw))
        io.log(f'--- done: left corner')
    return {'tree': features.tree, 'dlt': features.dlt, 'leftcorner': features.left_corner}


def parse_input(source):
    # TODO: collect input from various sources
    tokens = None
    try: 
        if Path(source).exists():
            raw = open(source, 'r').read()
            tokens = tokenize(raw)
        else:
            tokens = tokenize(source)
    except (FileNotFoundError, OSError) as e:
        tokens = tokenize(source)
    return tokens


def validate_input():
    # TODO: we just want a string
    pass


@path_decorator
def tokenize(raw):
    cmd = ['bash', 'tokenize.sh', raw.strip()]
    io.log(f'calling tokenizer like so: `{cmd}`')
    tokens = subprocess.check_output(cmd)
    io.log(f'---done--- tokenizer returned output like so: `{tokens}`')
    return tokens


@path_decorator
@cache_to_disk
def compute_trees(tokens, beam_width=5000):
    cmd = ['bash', 'parse_trees.sh', f'{beam_width}', tokens]
    trees = subprocess.check_output(cmd)
    return trees


@path_decorator
# @cache_to_disk
def compute_feature(feature, trees):
    cmd = ['bash', feature, trees]
    out = subprocess.check_output(cmd)
    return out
