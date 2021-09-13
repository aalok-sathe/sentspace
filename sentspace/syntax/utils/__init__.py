import os
from pathlib import Path
from urllib import request
import subprocess

from sentspace.utils import caching, text, wordnet, io


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
        os.chdir(str(target.parent))
        result = func(*args, **kwargs)
        os.chdir(previous_pwd)
        return result.decode('utf-8').strip()

    return wrapped



def get_content_ratio(is_content_pos_tag: tuple):
    """
    given boolean list corresponding to a token being a content word, 
    calculate the content ratio
    """
    return sum(is_content_pos_tag) / len(is_content_pos_tag)


def get_pronoun_ratio(pos_tags: tuple):
    """
    Given sentence calculate the pronoun ratio
    """
    pronoun_tags = {'PRP', 'PRP$', 'WP', 'WP$'}
    return sum(tag in pronoun_tags for tag in pos_tags) / len(pos_tags)


# @cache_to_mem
def get_is_content(taglst: tuple, content_pos=(wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV)):
    """
    Given list of POS tags, return list of 1 - content word, 0 - not content word
    """
    return tuple(int(text.get_wordnet_pos(tag) in content_pos) for tag in taglst)


@path_decorator
def tokenize(raw):
    cmd = ['bash', 'tokenize.sh', raw.strip()]
    # io.log(f'calling tokenizer like so: `{cmd}`')
    tokens = subprocess.check_output(cmd)
    # io.log(f'---done--- tokenizer returned output like so: `{tokens}`')
    return tokens


@path_decorator
@caching.cache_to_disk
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
def compute_feature(feature, trees):
    cmd = ['bash', feature, trees]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return out
