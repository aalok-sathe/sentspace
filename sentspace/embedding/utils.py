from collections import defaultdict
from contextlib import contextmanager
import os
import pickle
import typing
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

# import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer


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


def get_word_embeds(sentence: sentspace.Sentence.Sentence, w2v: typing.Dict[str, typing.Dict[str, np.array]], 
                    which: str = 'glove', dims: int = None) -> typing.Dict[str, typing.DefaultDict[None, list]]:
    """Extracts [static] word embeddings for tokens in the given sentence 

    Args:
        sentence ([sentspace.Sentence.Sentence]): a Sentence object
        w2v ([dict]): word embeddings dictionary as a mapping from token -> vector
        which (str, optional): [description]. Defaults to 'glove'.
        dims (int, optional): [description]. Defaults to 300.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # layer -> [emb_t1 emb_t2 emb_t3 ...]
    embeddings = defaultdict(list)
    
    dims = dims or next(iter(w2v[which].values())).shape[-1]

    # OOV_words = set()
    for token in sentence:
        if token in w2v[which]:
            embeddings[None].append(w2v[which][token])
        else:
            embeddings[None].append(np.repeat(np.nan, dims))
            # sentence.OOV[which].add(token)
            # OOV_words.add(token)
    
    return {which: embeddings}


@cache_to_mem
def load_huggingface(model_name_or_path: str = 'distilgpt2', device='cpu'):
    """loads and caches a huggingface model and tokenizer

    Args:
        model_name_or_path (str): Huggingface model hub identifier or path to directory 
                                  containing config and model weights. Defaults to 'gpt2'.

    Returns:
        Tuple[AutoModel, AutoTokenizer]: returns a model and a tokenizer
    """
    if 'TRANSFORMERS_CACHE' not in os.environ:
        os.environ['TRANSFORMERS_CACHE'] = str(Path(__file__).parent.parent.parent / 'TRANSFORMERS_CACHE/')
    io.log(f"loading HuggingFace model [{model_name_or_path}] using TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}")
    t = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=os.environ['TRANSFORMERS_CACHE'])
    m = AutoModel.from_pretrained(model_name_or_path, cache_dir=os.environ['TRANSFORMERS_CACHE'])
    m.to(device)
    m.eval()

    return m, t


def get_huggingface_embeds(sentence: sentspace.Sentence.Sentence, 
                           which: str = 'distilgpt2',
                           #layer: int = -1,
                           dims: int = None) -> typing.Dict[str, typing.DefaultDict[int, list]]:
    """Extracts [static] word embeddings for tokens in the given sentence 

    Args:
        sentence (sentspace.Sentence.Sentence): [description]
        which (str, optional): [description]. Defaults to 'distilgpt2'.
        dims (int, optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        typing.Dict[str, typing.DefaultDict[int, list]]: [description]
    """
    # layer -> [emb_t1 emb_t2 emb_t3 ...]
    embeddings = defaultdict(np.array)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_huggingface(which, device=device)

    n_layer = model.config.n_layer
    # max_n_tokens = model.config.n_positions

    # # [sentence_no -> (layer_no -> representation)]
    # reps = defaultdict(defaultdict(list))

    # we don't want to track gradients; only interested in encoding
    with torch.no_grad():

        # current procedure processes sentences individually. consider minibatching.
        batch_encoding = tokenizer(str(sentence), return_tensors="pt", truncation='longest_first').to(device)
        input_ids = batch_encoding['input_ids']
        
        overflow_tokens = max(0, len(input_ids) - model.config.n_positions)
        if overflow_tokens > 0: 
            io.log(f"Stimulus too long! Truncated the first {overflow_tokens} tokens", type='WARN')
        input_ids = input_ids[overflow_tokens:]
        
        # print(tokenizer.convert_ids_to_tokens(input_ids))

        output = model(input_ids, output_hidden_states=True, return_dict=True)

        hidden_states = output['hidden_states']

        for layer in range(len(hidden_states)):
            token = slice(None, None)
            rep = hidden_states[layer].detach().cpu().squeeze().numpy()[token, :]
            embeddings[layer] = rep 

        # print(input_ids.shape, rep.shape)

    return {which: embeddings}




def pool_sentence_embeds(sentence, token_embeddings, filters=[lambda i, x: True],
                         keys=None, methods={'mean', 'median'}):
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

    all_pooled = {} #defaultdict(lambda: defaultdict(dict))
    for which in token_embeddings:
        for layer in token_embeddings[which]:

            if keys and which not in keys: continue
            # all the embeddings corresponding to the tokens
            tokens = sentence.tokenized()
            all_embeds = [e for i, (t, e) in enumerate(zip(tokens, token_embeddings[which][layer]))]
            all_tokens = [t for i, (t, e) in enumerate(zip(tokens, token_embeddings[which][layer]))]
            all_embeds = np.array(all_embeds, dtype=np.float32)
            all_tokens = np.array(all_tokens, dtype=str)

            # print(all_tokens, all_embeds.shape)

            # exclude OOV words' embeddings (they are all NaNs)
            not_nan_tokens = all_tokens[~np.isnan(all_embeds[:, 0])   ]
            not_nan_embeds = all_embeds[~np.isnan(all_embeds[:, 0]), :]

            # make a note of the shape of the vector (n x embed_dim)
            shape = not_nan_embeds.shape
            # TODO: vectorize operation on all tokensnow that it is an numpy array
            if filters:
                mask = [all(fn(i, t) for filt_name, fn in filters.items()) for i, t in enumerate(not_nan_tokens)]
            else:
                mask = slice(None, None, None)

            filtered_embeds = not_nan_embeds[mask]
            filtered_shape = filtered_embeds.shape

            # if filtering left no tokens, we will use all_embeds instead
            if filtered_shape[0] == 0:
                io.log(f'filtered embeddings for current sentence are empty. retrying without filters: {tokens}', type='WARN')

                # now what? use unfiltered (as a fallback)
                filtered_embeds = not_nan_embeds
                
            # [very rarely] if no word has a corresponding embedding, then we have no choice
            # but to return a zero vector (or, sometime in the future, a random vector?)
            if shape[0] == 0:
                filtered_embeds = np.zeros((1, shape[-1]))


            pooled = defaultdict(lambda: defaultdict(dict))

            for method in methods:
                if type(method) is str:
                    method_name = method
                    if method_name == 'median':
                        pooled[layer][method_name][which] = np.median(filtered_embeds, axis=0).reshape(-1)#.tolist()
                    if method_name == 'mean':
                        pooled[layer][method_name][which] = filtered_embeds.mean(axis=0).reshape(-1)#.tolist()
                    if method_name == 'last':
                        pooled[layer][method_name][which] = filtered_embeds[-1, :].reshape(-1)#.tolist()
                    if method_name == 'first':
                        pooled[layer][method_name][which] = filtered_embeds[0, :].reshape(-1)#.tolist()
                elif type(method) is tuple:
                    method_name, fn = method
                    pooled[layer][method_name][which] = fn(filtered_embeds).reshape(-1)#.tolist()
                else:
                    raise ValueError(method)

                # 'pooled_'+which+'_max': filtered_embeds.max(axis=0).reshape(-1).tolist(),
                # 'pooled_'+which+'_min': filtered_embeds.min(axis=0).reshape(-1).tolist(),

            all_pooled.update(pooled)

    return all_pooled

