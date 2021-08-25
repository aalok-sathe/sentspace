import os
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib import request

from nltk.tree import ParentedTree
from sentspace.syntax import utils
from sentspace.syntax.features import DLT, Feature, LeftCorner, Tree
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk

os.environ['PERL_BADLANG'] = '0'



def get_features(sentence: str = None, dlt: bool = False, left_corner: bool = False) -> dict:
    """executes the syntactic features pipeline

    Args:
        sentence (str, optional): exactly 1 sentence [None].
        dlt (bool, optional): calculate DLT feature? [False].
        left_corner (bool, optional): calculate Left Corner feature? [False].

    Returns:
        sentspace.syntax.features.Feature: a Feature instance with appropriate attributes
    """
    features = Feature()
    if dlt or left_corner:
        # io.log(f'parsing into syntax tree: `{sentence}`')
        # parsed = parse_input(sentence)
        features.tree = Tree(compute_trees(sentence))
        # io.log(f'--- done: tree computed')
    else:
        return None

    # print(parse_input(sentence), features.tree)
    if dlt:
        # io.log(f'computing DLT feature')
        features.dlt = DLT(compute_feature('dlt.sh', features.tree.raw))
        # io.log(f'--- done: DLT')
    if left_corner:
        # io.log(f'computing left corner feature')
        features.left_corner = LeftCorner(compute_feature('leftcorner.sh', features.tree.raw))
        # io.log(f'--- done: left corner')

    tokenized = tuple(sentence.split())
    tagged_sentence = text.get_pos_tags(tokenized)
    is_content_word = utils.get_is_content(tagged_sentence, content_pos=text.pos_for_content)  # content or function word
    pronoun_ratio = utils.get_pronoun_ratio(tagged_sentence)
    content_ratio = utils.get_content_ratio(is_content_word)

    return {
        # 'UID': None,
        'sentence': sentence,
        # 'tags': tagged_sentence,

        # 'content_words': is_content_word,
        'pronoun_ratio': pronoun_ratio,
        'content_ratio': content_ratio,

        # 'tree': features.tree
        'dlt': features.dlt, 
        'leftcorner': features.left_corner,
    }



@utils.path_decorator
def tokenize(raw):
    cmd = ['bash', 'tokenize.sh', raw.strip()]
    io.log(f'calling tokenizer like so: `{cmd}`')
    tokens = subprocess.check_output(cmd)
    io.log(f'---done--- tokenizer returned output like so: `{tokens}`')
    return tokens


@utils.path_decorator
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


@utils.path_decorator
def compute_feature(feature, trees):
    cmd = ['bash', feature, trees]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return out
