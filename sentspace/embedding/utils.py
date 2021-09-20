import os
import pickle
import warnings
from pathlib import Path
import random

import numpy as np
import pandas as pd
import sentspace.utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.utils import Word, merge_lists, wordnet
from tqdm import tqdm


# --------- GloVe
def lowercase(f1g):
    """
    Return lowercase version of input (assume input is a list of token lists)
    """
    return [[token.lower() for token in sent] for sent in f1g]


def get_sent_version(version, df):
    """
    Return a list of sentences as lists of tokens given dataframe & version of token to use
    Options for version: 'raw', 'cleaned', 'lemmatized'
    """
    ref = {'raw': 'Word', 'cleaned': 'Word cleaned', 'lemmatized': 'Word lemma'}
    version = ref[version]
    f1g = []
    for i in df['Sentence no.'].unique():
        f1g.append(list(df[df['Sentence no.'] == i].sort_values('Word no. within sentence')[version]))
    return f1g


def get_vocab(token_list):
    """
    Return set of unique tokens in input (assume input is a list of tokens)
    """
    return set(t for t in token_list)


def download_embeddings(which='glove.840B.300d.txt'):
    raise NotImplementedError()
    if 'glove' in which:
        url = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip'


@cache_to_mem
def load_embeddings(emb_file: str = 'glove.840B.300d.txt',
                    data_dir: Path = None,
                    vocab: tuple = ()):
    """
    Read through the embedding file to find embeddings for target words in vocab
    Return dict mapping word to embedding (numpy array)
    """
    try:
        data_dir = Path(data_dir)
    except TypeError:
        data_dir = Path(__file__).parent / '..' / '..' / '.feature_database/'
    
    vocab = set(vocab)
    OOV = set(vocab)

    io.log(f"loading embeddings from {emb_file} for vocab of size {len(vocab)}")
    w2v = {}
    with (data_dir / emb_file).open('r') as f:
        total_lines = sum(1 for _ in tqdm(f, desc=f'counting # of lines in {data_dir/emb_file}'))
    with (data_dir / emb_file).open('r') as f:
        for line in tqdm(f, total=total_lines, desc=f'searching for embeddings in {emb_file}'):
            token, *emb = line.split(' ')
            if token in vocab:
                # print(f'found {token}!')
                w2v[token] = np.asarray(emb, dtype=float)
                OOV.remove(token)
    
    io.log(f"---done--- loading embeddings from {emb_file}. OOV count: {len(OOV)}/{len(vocab)}")
    OOVlist = [*OOV]
    random.shuffle(OOVlist)
    io.log(f"           a selection of up to 100 random OOV tokens: {OOVlist[:100]}")

    return w2v


def get_word_embeds(token_list, w2v, which='glove', dims=300, return_NA_words=False, save=False, save_path=False):
    """[summary]

    Args:
        tokenized ([type]): [description]
        w2v ([type]): [description]
        which (str, optional): [description]. Defaults to 'glove'.
        dims (int, optional): [description]. Defaults to 300.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """    
    #
    """
    Return dataframe of each word, sentence no., and its glove embedding
    If embedding does not exist for a word, fill cells with np.nan
    Parameters:
        f1g: list of sentences as lists of tokens
        w2v: dict mapping word to embedding
        return_NA_words: optionally return unique words that are NA
        save: whether to save results
        save_path: path to save, support .csv & .mat files
    """

    embeddings = []
    
    OOV_words = set()
    for token in token_list:
        if token in w2v[which]:
            embeddings.append(w2v[which][token])
        else:
            embeddings.append([np.nan]*dims)
            OOV_words.add(token)
    
    return embeddings



def pool_sentence_embeds(tokens, token_embeddings, filters=[lambda i, x: True],
                         which='glove'):
    """pools embeddings of an entire sentence (given as a list of embeddings)
       using averaging, maxpooling, minpooling, etc., after applying all the
       provided filters as functions (such as content words only).

    Args:
        token_embeddings (list[np.array]): [description]
        filters (list[function[(idx, token) -> bool]], optional): [description]. Defaults to [lambda x: True].
            filters should be functions that map token to bool (e.g. is_content_word(...))
            only tokens that satisfy all filters are retained.

    Returns:
        dict: averaging method -> averaged embedding
    """                         

    """
    Return dataframe of each sentence no. and its sentence embedding
    from averaging embeddings of words in a sentence (ignore NAs)
    Parameters:
        df: dataframe, output of get_glove_word()
        content_only: if True, use content words only
        is_content_lst: list, values 1 if token is content word, 0 otherwise
        save: whether to save results
        save_path: path to save, support .csv & .mat files
    """
    # if content_only:
    #     df = df[np.array(is_content_lst) == 1]

    all_pooled = {}
    for which in token_embeddings:
        # all the embeddings corresponding to the tokens
        all_embeds = [e for i, (t, e) in enumerate(zip(tokens, token_embeddings[which]))]
        all_tokens = [t for i, (t, e) in enumerate(zip(tokens, token_embeddings[which]))]
        all_embeds = np.array(all_embeds, dtype=np.float32)
        all_tokens = np.array(all_tokens, dtype=str)

        # exclude OOV words' embeddings (as they are all NaNs)
        # make a note of the shape of the vector (n x embed_dim)
        not_nan_tokens = all_tokens[~np.isnan(all_embeds[:, 0])   ]
        not_nan_embeds = all_embeds[~np.isnan(all_embeds[:, 0]), :]

        shape = not_nan_embeds.shape
        # TODO: vectorize operation on all tokensnow that it is an numpy array
        mask = [all(fn(i, t) for fn in filters) for i, t in enumerate(not_nan_tokens)]

        filtered_embeds = not_nan_embeds[mask]
        filtered_shape = filtered_embeds.shape

        # if filtering left no tokens, we will use all_embeds instead
        if filtered_shape[0] == 0:
            io.log(f'all embeddings for current sentence are NaN ({shape} -> {filtered_embeds.shape}): {tokens}', type='WARN')
            
            # now what? use all_embeds (as a fallback)
            filtered_embeds = not_nan_embeds
            
        # [very rarely] if no word has a corresponding embedding, then we have no choice
        # but to return a zero vector (or, sometime in the future, a random vector?)
        if shape[0] == 0:
            filtered_embeds = np.zeros((1, shape[-1]))


        pooled = {
            'pooled_'+which+'_median': np.median(filtered_embeds, axis=0).reshape(-1).tolist(),
            'pooled_'+which+'_mean': filtered_embeds.mean(axis=0).reshape(-1).tolist(),
            # 'pooled_'+which+'_max': filtered_embeds.max(axis=0).reshape(-1).tolist(),
            # 'pooled_'+which+'_min': filtered_embeds.min(axis=0).reshape(-1).tolist(),
        }

        all_pooled.update(pooled)

    return all_pooled

