from collections import defaultdict
import os
import sentspace
from pathlib import Path

import pandas as pd
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


def get_features(sentence: str = None, identifier: str = None, 
                 dlt: bool = True, left_corner: bool = True,
                 syntax_port: int = 8000) -> dict:
    """executes the syntactic features pipeline

    Args:
        sentence (str, optional): exactly 1 sentence [None].
        dlt (bool, optional): calculate DLT feature? [False].
        left_corner (bool, optional): calculate Left Corner feature? [False].

    Returns:
        sentspace.syntax.features.Feature: a Feature instance with appropriate attributes
    """

    # if the "sentence" actually consists of multiple sentences (as determined by the 
    # NLTK Punkt Sentence Tokenizer), we want to repeat the below block per sentence and 
    # then pool across sentences
    from nltk.tokenize import sent_tokenize 
    stripped = ''.join([i if ord(i) < 128 else '' for i in sentence])
    sentences = sent_tokenize(stripped, language='english')
    features_to_pool = defaultdict(list)
    features = None
    dlt_concat, left_corner_concat = None, None

    for i, sub_sentence in enumerate(sentences):

        features = Feature()
        if dlt or left_corner:
            # io.log(f'parsing into syntax tree: `{sentence}`')
            # parsed = parse_input(sentence)
            try:
                server_url = f'http://localhost:{syntax_port}/fullberk'
                features.tree = Tree(utils.compute_trees(sub_sentence, server_url=server_url)) 
                if type(features.tree) is RuntimeError:
                    raise features.tree

                getattr(features.tree, 'raw') 
                # print(parse_input(sentence), features.tree)
                if dlt and features.tree.raw is not None:
                    # io.log(f'computing DLT feature')
                    dlt_stdout = utils.compute_feature('dlt.sh', features.tree.raw)
                    if type(dlt_stdout) == RuntimeError:
                        raise dlt_stdout
                    else:
                        features.dlt = DLT(dlt_stdout, sub_sentence, identifier)
                    # io.log(f'--- done: DLT')
                if left_corner and features.tree.raw is not None:
                    # io.log(f'computing left corner feature')
                    left_corner_stdout = utils.compute_feature('leftcorner.sh', features.tree.raw)
                    if type(left_corner_stdout) == RuntimeError:
                        raise left_corner_stdout
                    else:
                        features.left_corner = LeftCorner(left_corner_stdout, sub_sentence, identifier)
                    # io.log(f'--- done: left corner')

                features_to_pool['dlt'] += [features.dlt]
                features_to_pool['left_corner'] += [features.left_corner]

            except AttributeError as ae:
                import traceback
                io.log(f'FAILED: AttributeError while processing '
                       f'Tree [{features.tree}] features for chunk [{sub_sentence}] of sentence [{sentence}] '
                       f'traceback: {traceback.format_exc()}', 
                       type='ERR')
                # for attr in ['dlt', 'left_corner', 'tree']:
                #     io.log(f'hasattr(features, {attr}): {hasattr(features, attr)}', type='ERR')
                # io.log(f'hasattr(features.tree, raw): {hasattr(features.tree, "raw")}', type='ERR')
                pass

            except RuntimeError:
                io.log(f'FAILED: RuntimeError to process Tree features for chunk [{sub_sentence}] of sentence [{sentence}]', type='ERR')
                pass

         

            # do the groupby index and mean() here and then continue
            if dlt:
                try:
                    dlt_concat = pd.concat(features_to_pool['dlt'], axis='index')
                    dlt_concat = dlt_concat.groupby('index').mean()
                    dlt_concat['sentence'] = sentence
                except ValueError:
                    dlt_concat = pd.DataFrame()
            else:
                dlt_concat = None
            if left_corner:
                try:
                    left_corner_concat = pd.concat(features_to_pool['left_corner'], axis='index')
                    left_corner_concat = left_corner_concat.groupby('index').mean()
                    left_corner_concat['sentence'] = sentence
                except ValueError:
                    left_corner_concat = pd.DataFrame()
            else:
                left_corner_concat = None    



    tokenized = utils.tokenize(sub_sentence).split()
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
        'dlt': dlt_concat, 
        'leftcorner': left_corner_concat,
    }


