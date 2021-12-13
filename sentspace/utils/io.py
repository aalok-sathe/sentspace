
from genericpath import exists
import os
import pickle
import textwrap
import datetime # from datetime import date
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
    """Creates an output directory structure to output the results of the pipeline based on the supplied input file

    Args:
        input_file (str): [description]
        output_dir (str): [description]
        calling_module ([type], optional): [description]. Defaults to None.
        stop_words_file (str, optional): [description]. Defaults to None.

    Returns:
        Path: [description]
    """    
    output_dir = Path(output_dir)
    # output will be organized based on its respective input file,
    # together with a hash of the contents (for to avoid the case where two files are 
    # named similarly to one another but contain different contents)
    out_file_name = '.'.join(os.path.basename(input_file).split('.')[:-1])

    output_dir /= out_file_name
    output_dir /= sentspace.utils.md5(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / 'md5.txt').open('w') as f:
        f.write(f'{Path(input_file).resolve()} md5sum:\t' + sentspace.utils.md5(input_file))
    with (output_dir / 'run_history.txt').open('a+') as f:
        f.write(datetime.datetime.now().strftime('run: %Y-%m-%d %H:%M:%S %Z\n'))

    output_dir /= calling_module or '' # to subdir it by "lexical" or "syntax" etc.
    # output_dir /= date.today().strftime('run_%Y-%m-%d')
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


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

    if stop_words_file:
        raise NotImplementedError()
        stop_words = set(np.loadtxt(stop_words_file, delimiter='\t', unpack=False, dtype=str))

    if filename.endswith('.txt'):
        UIDs = []
        sentences = []
        with open(filename, 'r') as f:
            UID_prefix = f'{filename[-8:]:#>10}' + '_' + sentspace.utils.md5(filename)[-5:]
            for i, line in enumerate(f):

                uid = UID_prefix + '_' + f'{len(UIDs):0>5}'
                UIDs += [uid]
                
                s = Sentence(line, uid)
                if s: 
                    sentences.append(s)

        return sentences

    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename, sep=',')
    elif filename.endswith('.tsv'):
        df = pd.read_csv(filename, sep='\t')
    else:
        raise ValueError('unknown type of file supplied (must be txt/pkl/csv/tsv. if pickle, must be a dataframe object)')

    # try to figure out what to use as a unique identifier for the sentence
    if 'corpora_identifier' in df.columns:
        UIDs = df['corpora_identifier'].tolist()
    elif 'index' in df.columns:
        UIDs = df['index'].tolist()
    else:
        try: 
            UIDs = df.index.tolist()
        except AttributeError:
            raise('does your dataframe have a unique index for each sentence?')

    sentences = [Sentence(raw, uid) for raw, uid in zip(df['sentence'].tolist(), UIDs)]
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
