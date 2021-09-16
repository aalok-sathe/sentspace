import os
import sentspace
from pathlib import Path

from nltk.tree import ParentedTree
from sentspace.syntax import utils
from sentspace.syntax.features import DLT, Feature, LeftCorner, Tree
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk

__pdoc__ = {'compute_tree_dlt_left_corner': False,
            'utils.calcEmbd': False,
            'utils.calcDLT': False,
            'utils.printlemmas': False,
            'utils.tree': False
            }

os.environ['PERL_BADLANG'] = '0'


def get_features(sentence: str = None, identifier: str = None, dlt: bool = True, left_corner: bool = True) -> dict:
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
        features.tree = Tree(utils.compute_trees(sentence))
        # io.log(f'--- done: tree computed')
    else:
        return None

    # print(parse_input(sentence), features.tree)
    if dlt:
        # io.log(f'computing DLT feature')
        features.dlt = DLT(utils.compute_feature('dlt.sh', features.tree.raw), sentence, identifier)
        # io.log(f'--- done: DLT')
    if left_corner:
        # io.log(f'computing left corner feature')
        features.left_corner = LeftCorner(utils.compute_feature('leftcorner.sh', features.tree.raw), sentence, identifier)
        # io.log(f'--- done: left corner')

    tokenized = utils.tokenize(sentence).split()
    tagged_sentence = text.get_pos_tags(tokenized)
    is_content_word = utils.get_is_content(tagged_sentence, content_pos=text.pos_for_content)  # content or function word
    pronoun_ratio = utils.get_pronoun_ratio(tagged_sentence)
    content_ratio = utils.get_content_ratio(is_content_word)

    return {
        'index': identifier,
        'sentence': sentence,
        # 'tags': tagged_sentence,

        # 'content_words': is_content_word,
        'pronoun_ratio': pronoun_ratio,
        'content_ratio': content_ratio,

        # 'tree': features.tree
        'dlt': features.dlt, 
        'leftcorner': features.left_corner,
    }


