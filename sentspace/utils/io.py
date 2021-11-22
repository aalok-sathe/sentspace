
import os
import pickle
import textwrap
from datetime import date
from hashlib import sha1
from pathlib import Path
from sys import stderr, stdout
import pandas as pd
import numpy as np
import sentspace.utils
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.s3 import load_feature
from tqdm import tqdm
from sentspace.Sentence import Sentence


# from sentspace.utils.sanity_checks import sanity_check_databases

def dump_features(): pass

def create_output_paths(input_file:str, output_dir:str, calling_module=None, stop_words_file:str=None) -> Path:
    embed_method = 'all'  # options: 'strict', 'all'
    content_only = False

    output_dir = Path(output_dir)

    suffix = ''

    # output will be organized based on its respective input file,
    # together with a hash of the contents (for completeness)
    out_file_name = os.path.basename(input_file).split('.')[0]
    # with open(input_file, 'r') as f:
    output_dir /= (out_file_name + '_md5:' + sentspace.utils.md5(input_file))
    output_dir /= calling_module or ''
    # output_dir /= date.today().strftime('run_%Y-%m-%d')
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

    sent_suffix = f"_{embed_method}"

    if content_only:
        sent_suffix = '_content' + sent_suffix
    if stop_words_file:
        sent_suffix = '_content'+'_minus_stop_words' + sent_suffix
    # Make out outlex path
    word_lex_output_path, embed_lex_output_path, plot_path, na_words_path, bench_perc_out_path = _create_output_paths(
        output_dir, out_file_name, 'lexical', sent_suffix=sent_suffix)
    # Make output syntax path
    _, sent_output_path = _create_output_paths(
        output_dir, out_file_name, 'syntax', sent_suffix=sent_suffix)
    glove_words_output_path, glove_sents_output_path = _create_output_paths(
        output_dir, out_file_name, 'embedding', sent_suffix=sent_suffix)
    pmi_paths = _create_output_paths(
        output_dir, out_file_name, 'PMI', sent_suffix=sent_suffix)

    # lex_base = f'analysis_example/{date_}\\lex\\'

    return sent_output_path, glove_words_output_path, glove_sents_output_path


# def _create_output_paths(output_dir:Path, name, analysis, suffix='', sent_suffix=''):
#     """
#     Return list of file paths and create output folder if appropriate
#     Supports analysis = 'lex', 'glove','syntax','PMI'
#     """
#     if analysis == 'lexical':
#         return_paths = [f"{name}_lex_features_words{suffix}.csv",
#                         f"{name}_lex_features_sents{suffix}{sent_suffix}.csv",
#                         f"{name}_plots{suffix}{sent_suffix}_",
#                         f"{name}_unique_NA{suffix}{sent_suffix}.csv",
#                         f"{name}_benchmark_percentiles{suffix}{sent_suffix}.csv"]
#     elif analysis == 'embedding':
#         return_paths = [f"{name}_glove_words{suffix}.csv",
#                         f"{name}_glove_sents{suffix}{sent_suffix}.csv"]
#     elif analysis == 'PMI':
#         return_paths = [f"{name}_pPMI_0{suffix}.csv",
#                         f"{name}_pPMI_1{suffix}.csv",
#                         f"{name}_pPMI_2{suffix}.csv",
#                         f"{name}_ngrams{suffix}.pkl",
#                         f"{name}_nm1grams{suffix}.pkl"]
#     elif analysis == 'syntax':
#         return_paths = [f"{name}_{suffix}.csv",
#                         f"{name}_{suffix}{sent_suffix}.csv"]
#     elif analysis == 'lex_per_word':
#         return_paths = [f"{name}_{suffix}.csv",
#                         f"{name}_{suffix}{sent_suffix}.csv"]
#     else:
#         raise ValueError('Invalid analysis method!')
#     output_dir = os.path.join(output_dir, analysis)
#     if not os.path.isdir(output_dir):  # create output_folder if it doesn't exist
#         Path(output_dir).mkdir(parents=True, exist_ok=True)

#     result = [os.path.join(output_dir, path) for path in return_paths]
#     return result


def read_sentences(filename: str, stop_words_file: str = None):
    """reads sentences from a file, one per line, filtering
        for stopwords

    Args:
        filename (str): path to input file containing sentences
        stop_words_file (str, optional): path to file containing stopwords

    Returns:
        list[list]: list of the tokens from each sentences, nested as a list
        list: list of sentences
    """
    

    # if a non-empty collection of stop words has been supplied
    if stop_words_file:
        raise NotImplementedError()
        stop_words = set(np.loadtxt(stop_words_file, delimiter='\t', unpack=False, dtype=str))

    if filename.endswith('.txt'):
        UIDs = []
        token_lists = []
        sentences = []
        with open(filename, 'r') as f:
            UID_prefix = f'{filename[-8:]:#>10}' + '_' + sentspace.utils.md5(filename)[-5:]
            for i, line in enumerate(f):

                uid = UID_prefix + '_' + f'{len(UIDs):0>5}'
                UIDs += [uid]
                
                s = Sentence(line, uid)

                # if line.strip():
                #     tokens = sentspace.utils.text.tokenize(line) # line.split()
                if s: 
                    sentences.append(s)

                # token_lists.append(tokens)

        # return UIDs, token_lists, sentences
        return sentences

    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename, sep=',')
    elif filename.endswith('.tsv'):
        df = pd.read_csv(filename, sep='\t')
    else:
        raise ValueError('unknown type of file supplied (must be txt/pkl/csv/tsv. if pickle, must be a dataframe object)')

    try:
        UIDs = df['corpora_identifier']
        #df['index'].tolist()
    except KeyError:
        UIDs = df.index.tolist()
    except AttributeError:
        raise('does your dataframe have a unique index for each sentence?')

    sentences = [Sentence(raw, uid) for raw, uid in zip(df['sentence'].tolist(), UIDs)]
    # token_lists = df['sentence'].apply(lambda s: sentspace.utils.text.tokenize(s)).tolist()

    # return UIDs, token_lists, sentences
    return sentences


def get_batches(iterable, batch_size:int):
    """
    splits iterable into batches of size batch_size
    """
    batch = []
    for i, item in enumerate(iterable):
        batch.append(item)
        if (i + 1) % batch_size == 0:
            yield batch
            batch = []
    if batch:
        yield batch


def load_surprisal(file='pickle/surprisal-3_dict.pkl'):
    """
    Load dict mapping word to surprisal value
    """
    with open(file, 'rb') as f:
        return pickle.load(f)


def log(message, type='INFO'):
    
    class T:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    if type == 'INFO':
        c = T.OKCYAN
    elif type == 'EMPH':
        c = T.OKGREEN
    elif type == 'WARN':
        c = T.BOLD + T.WARNING
    elif type == 'ERR':
        c = '\n' + T.BOLD + T.FAIL
    else:
        c = T.OKBLUE

    timestamp = f'{sentspace.utils.time() - sentspace.utils.START_TIME():.2f}s'
    lines = textwrap.wrap(message+T.ENDC,
                          width=120, 
                          initial_indent = c + '%'*4 + f' [{type} @ {timestamp}] ', 
                          subsequent_indent='.'*20+' ')
    tqdm.write('\n'.join(lines), file=stderr)
    # print(*lines, sep='\n', file=stderr)


# def compile_token_dicts() -> pd.DataFrame:
