import os
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib import request

from nltk.tree import ParentedTree
from sentspace.syntax.features import DLT, Feature, LeftCorner, Tree
from sentspace.utils import io
from sentspace.utils.caching import cache_to_disk

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
        result = func(*args, **kwargs)
        os.chdir(previous_pwd)
        return result.decode('utf-8').strip()

    return wrapped


def get_features(text: str = None, dlt: bool = False, left_corner: bool = False):
    """executes the syntactic features pipeline

    Args:
        text (str, optional): exactly 1 sentence [None].
        dlt (bool, optional): calculate DLT feature? [False].
        left_corner (bool, optional): calculate Left Corner feature? [False].

    Returns:
        sentspace.syntax.features.Feature: a Feature instance with appropriate attributes
    """
    features = Feature()
    if dlt or left_corner:
        # io.log(f'parsing given text')
        # parsed = parse_input(text)
        io.log(f'computing tree for parsed text')
        features.tree = Tree(compute_trees(text))
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


# def parse_input(source):
#     # TODO: collect input from various sources
#     tokens = None
#     try:
#         if Path(source).exists():
#             raw = open(source, 'r').read()
#             tokens = tokenize(raw)
#         else:
#             tokens = tokenize(source)
#     except (FileNotFoundError, OSError) as e:
#         tokens = tokenize(source)
#     return tokens


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
def compute_trees(sentence, server_url='http://localhost:8000/fullberk'):
    
    data = f'{{ "sentence": "{sentence}" }}'
    r = request.Request(server_url, data=bytes(data, 'utf-8'), method='GET',
                        headers={'Content-Type': 'application/json'})
    with request.urlopen(r) as rq:
        response = rq.read()
    cmd = ['bash', 'postprocess_trees.sh', response]

    # fallback to manually initializing parser
    # cmd = ['bash', 'parse_trees.sh', tokens]
    trees = subprocess.check_output(cmd)
    return trees


@path_decorator
# @cache_to_disk
def compute_feature(feature, trees):
    cmd = ['bash', feature, trees]
    out = subprocess.check_output(cmd)
    return out
