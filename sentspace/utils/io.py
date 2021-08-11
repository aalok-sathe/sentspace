
import os
from pathlib import Path
import pickle
from datetime import date
from hashlib import sha1

import numpy as np
import sentspace.utils
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.s3 import load_feature
from sentspace.utils.sanity_checks import sanity_check_databases


def create_output_paths(input_file:str, output_dir:str, calling_module=None, stop_words_file:str=None):
    embed_method = 'all'  # options: 'strict', 'all'
    content_only = False

    output_dir = Path(output_dir)

    suffix = ''

    # output will be organized based on its respective input file,
    # together with a hash of the contents (for completeness)
    out_file_name = os.path.basename(input_file).split('.')[0]
    with open(input_file, 'r') as f:
        output_dir /= (out_file_name + '_' + sentspace.utils.sha1(f.read()))
    output_dir /= calling_module or ''
    output_dir /= date.today().strftime('%Y-%m-%d')
    output_dir.mkdir(parent=True, exist_ok=True)

    sent_suffix = f"_{embed_method}"

    if content_only:
        sent_suffix = '_content' + sent_suffix
    if stop_words_file:
        sent_suffix = '_content'+'_minus_stop_words' + sent_suffix
    # Make out outlex path
    word_lex_output_path, embed_lex_output_path, plot_path, na_words_path, bench_perc_out_path = _create_output_paths(
        output_folder, out_file_name, 'lex', sent_suffix=sent_suffix)
    # Make output syntax path
    _, sent_output_path = _create_output_paths(
        output_folder, out_file_name, 'syntax', sent_suffix=sent_suffix)
    glove_words_output_path, glove_sents_output_path = _create_output_paths(
        output_folder, out_file_name, 'glove', sent_suffix=sent_suffix)
    pmi_paths = _create_output_paths(
        output_folder, out_file_name, 'PMI', sent_suffix=sent_suffix)

    lex_base = f'analysis_example/{date_}\\lex\\'

    return sent_output_path, glove_words_output_path, glove_sents_output_path


def _create_output_paths(output_dir:Path, name, analysis, suffix='', sent_suffix=''):
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
        Path(output_folder).mkdir(parents=True, exist_ok=True)

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
