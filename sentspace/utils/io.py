
import os
import pathlib
import pickle

import numpy as np
import sentspace.utils
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.s3 import load_feature
from sentspace.utils.sanity_checks import sanity_check_databases


def create_output_path(output_folder, name, analysis, suffix='', sent_suffix=''):
    """
    Return list of file paths and create output folder if appropriate
    Supports analysis = 'lex', 'glove','syntax','PMI'
    """
    if analysis == 'lex':
        return_paths = [f"{name}_lex_features_words{suffix}.csv",
                        f"{name}_lex_features_sents{suffix}{sent_suffix}.csv",
                        f"{name}_plots{suffix}{sent_suffix}_",
                        f"{name}_unique_NA{suffix}{sent_suffix}.csv",
                        f"{name}_benchmark_percentiles{suffix}{sent_suffix}.csv"]
    elif analysis == 'glove':
        return_paths = [f"{name}_glove_words{suffix}.csv",
                        f"{name}_glove_sents{suffix}{sent_suffix}.csv"]
    elif analysis == 'PMI':
        return_paths = [f"{name}_pPMI_0{suffix}.csv",
                        f"{name}_pPMI_1{suffix}.csv",
                        f"{name}_pPMI_2{suffix}.csv",
                        f"{name}_ngrams{suffix}.pkl",
                        f"{name}_nm1grams{suffix}.pkl"]
    elif analysis == 'syntax':
        return_paths = [f"{name}_{suffix}.csv",
                        f"{name}_{suffix}{sent_suffix}.csv"]
    elif analysis == 'lex_per_word':
        return_paths = [f"{name}_{suffix}.csv",
                        f"{name}_{suffix}{sent_suffix}.csv"]
    else:
        raise ValueError('Invalid analysis method!')
    output_folder = os.path.join(output_folder, analysis)
    if not os.path.isdir(output_folder):  # create output_folder if it doesn't exist
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    result = [os.path.join(output_folder, path) for path in return_paths]
    return result


def read_sentences(filename, stop_words_file: str = None):
    """reads sentences from a file, one per line, filtering
        for stopwords

    Args:
        filename (str): path to input file containing sentences
        stop_words_file (str, optional): path to file containing stopwords

    Returns:
        list[list]: list of the tokens from each sentences, nested as a list
        list: list of sentences
    """

    token_lists = []
    sentences = []

    with open(filename, 'r') as f:
        for line in f:
            tokens = line.split()
            # if a non-empty collection of stop words has been supplied
            if stop_words_file:
                stop_words = np.loadtxt(
                    stop_words_file, delimiter='\t', unpack=False, dtype=str)
                stop_words = set(stop_words)
                filtered = [x for x in tokens if x not in stop_words]
                token_lists.append(filtered)
                sentences.append(' '.join(filtered))
            # no stopwords supplied; do not filter
            else:
                token_lists.append(tokens)
                sentences.append(line)

    return token_lists, sentences


@cache_to_mem
def load_databases(features='all', path='.feature_database/', ignore_case=True):
    """
    Load dicts mapping word to feature value
    If one feature, provide in list format
    """
    databases = {}
    if features == 'all':
        features = sentspace.utils.get_feature_list()
    for feature in features:
        if not os.path.exists(path+feature+'.pkl'):
            load_feature(key=feature+'.pkl')
        with open(path+feature+'.pkl', 'rb') as f:
            d = pickle.load(f)
            if ignore_case:  # add lowercase version to feature database
                for key, val in d.copy().items():
                    d[str(key).lower()] = val
            databases[feature] = d

    sanity_check_databases(databases)
    return databases




def load_surprisal(file='pickle/surprisal-3_dict.pkl'):
    """
    Load dict mapping word to surprisal value
    """
    with open(file, 'rb') as f:
        return pickle.load(f)
