import os
from pathlib import Path
from typing import List
import copy

import sentspace.utils
from sentspace.lexical import utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.resources import feat_rename_dict


def get_features(sentence: sentspace.Sentence.Sentence, lock=None) -> dict:

    # io.log(f'computing lexical featuures for `{sentence}`')

    # if lock: lock.acquire()
    databases = utils.load_databases(features="all")
    # if lock: lock.release()

    features_from_database = utils.get_all_features(
        sentence, databases
    )  # lexical features

    # Rename keys in features_from_database if they exist in feat_rename_dict
    for key, val in features_from_database.copy().items():
        if key in feat_rename_dict:
            features_from_database[feat_rename_dict[key]] = features_from_database.pop(
                key
            )

    accumulator = []
    # return list of token-level features, as a dict per token
    for i, token in enumerate(sentence.tokens):
        db_features_slice = {
            feature: features_from_database[feature][i]
            for feature in features_from_database
        }

        accumulator += [{
            "index": sentence.uid,
            "sentence": str(sentence),
            "token": token,
            "lemma": sentence.lemmas[i],
            "tag": sentence.pos_tags[i],
            "content_word": sentence.content_words[i],
            **db_features_slice,
        }]

    return accumulator
