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
from sentspace.utils.caching import cache_to_mem #, cache_to_disk
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer


def download_embeddings(model_name='glove.840B.300d.txt'):
    raise NotImplementedError
    if 'glove' in model_name:
        url = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip'
        # download(url)




def flatten_activations(activations: typing.Dict[int, np.array]) -> pd.DataFrame:
    """
    Convert layer-wise activations into flattened dataframe format.
    Input: dict, key = layer, item = nd array of representations of that layer (n_tokens, )
    Output: pd dataframe, MultiIndex (layer, unit)
    """
    labels = []
    arr_flat = []
    for layer, act_arr in activations.items():
        arr_flat.append(act_arr.reshape(1,-1))
        for i in range(act_arr.shape[0]): # across units
            labels.append(('representation', layer, i,))
    arr_flat = np.concatenate(arr_flat, axis=1) # concatenated activations across layers
    df = pd.DataFrame(arr_flat)
    df.columns = pd.MultiIndex.from_tuples(labels) # rows: stimuli, columns: units
    return df


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
            if token in vocab or len(vocab) == 0:
                # print(f'found {token}!')
                w2v[token] = np.asarray(emb, dtype=float)
                OOV.difference_update({token}) # calling .remove() on an empty set would give an error
    
    io.log(f"---done--- loading embeddings from {emb_file}. OOV count: {len(OOV)}/{len(vocab)}")
    OOVlist = [*OOV]
    random.shuffle(OOVlist)
    io.log(f"           a selection of up to 32 random OOV tokens: {OOVlist[:32]}")

    return w2v


def get_word_embeds(sentence: sentspace.Sentence.Sentence, w2v: typing.Dict[str, np.array], 
                    model_name: str = 'glove', dims: int = None) -> typing.Dict[str, typing.DefaultDict[None, list]]:
    """Extracts [static] word embeddings for tokens in the given sentence 

    Args:
        sentence ([sentspace.Sentence.Sentence]): a Sentence object
        w2v ([dict]): word embeddings dictionary as a mapping from token -> vector
        model_name (str, optional): [description]. Defaults to 'glove'.
        dims (int, optional): [description]. Defaults to 300.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # layer -> [emb_t1 emb_t2 emb_t3 ...]
    embeddings = defaultdict(list)
    dims = dims or next(iter(w2v[model_name].values())).shape[-1]

    # OOV_words = set()
    for token in sentence:
        if token in w2v:
            embeddings[0].append(w2v[token])
        else:
            embeddings[0].append(np.repeat(np.nan, dims))
            # sentence.OOV[model_name].add(token)
            # OOV_words.add(token)
    
    return {model_name: embeddings}


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
                           model_name: str = 'distilgpt2',
                           layers: typing.Collection[int] = None,
                           dims: int = None) -> typing.Dict[str, typing.DefaultDict[int, list]]:
    """Extracts [static] word embeddings for tokens in the given sentence 

    Args:
        sentence (sentspace.Sentence.Sentence): [description]
        model_name (str, optional): [description]. Defaults to 'distilgpt2'.
        layers (list[int], optional): a collection of layers to extract from the model. if None, all layers are extracted.
        dims (int, optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        typing.Dict[str, typing.DefaultDict[int, list]]: [description]
    """
    # layer -> [emb_t1 emb_t2 emb_t3 ...]
    representations = dict()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_huggingface(model_name, device=device)

    # we don't want to track gradients; only interested in the encoding
    model.eval()
    with torch.no_grad():

        # current procedure processes sentences individually. consider minibatching.
        batch_encoding = tokenizer(str(sentence), return_tensors="pt", truncation='longest_first').to(device)
        input_ids = batch_encoding['input_ids']
        
        # overflow_tokens = max(0, len(input_ids) - model.config.n_positions)
        # if overflow_tokens > 0: io.log(f"Stimulus too long! Truncated the first {overflow_tokens} tokens", type='WARN')
        
        # input_ids = input_ids[overflow_tokens:]
        # print(tokenizer.convert_ids_to_tokens(input_ids))

        output = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = output['hidden_states']

        #  for i in range(n_layer+1):
        for layer in range(len(hidden_states)):
            if layers is None or layer in layers:
                token = slice(None, None) # placeholder to allow a possibility of picking a particular token rather than the full sequence
                representations[layer] = hidden_states[layer].detach().cpu().squeeze().numpy()[token, :]

        # print(input_ids.shape, representations[0].shape)

    return {model_name: representations}

    

def pool_sentence_embeds(sentence, token_embeddings, filters={'nofilter': lambda i, x: True},
                         keys=None, methods={'mean', 'median'}):
    """pools embeddings of an entire sentence (given as a list of embeddings)
       using averaging, maxpooling, minpooling, etc., after applying all the
       provided filters as functions (such as content words only).

    Args:
        token_embeddings (list[np.array]): [description]
        filters (list[function[(idx, token) -> bool]], optional): [description]. Defaults to [lambda x: True].
            filters should be functions that map token to bool (e.g. is_content_word(...))
            only tokens that satisfy all filters are retained.
        keys (`typing.Union[typing.Collection[str], None]`): which models we want to pool using the methods supplied. 
            if None, all available models are pooled (separately) using the supplied methods
        methods (`typing.Collection[typing.Union[str, typing.Tuple[str, typing.Callable]]]`):

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
    
    # model -> method -> repr
    all_pooled = defaultdict(dict)

    for model_name in token_embeddings:

        # if the current model_name is not meant to be aggregated in this manner, skip
        # (e.g. BERT and last token "aggregation")
        # if keys is None, any aggregation step specified will be applied regardless of model_name
        # (e.g., mean)
        if keys and model_name not in keys: 
            continue

        # map from method --> layer (int) --> pooled representation per model
        # this entity still needs to be flattened
        model_pooled = defaultdict(dict)

        for layer in token_embeddings[model_name]:

            # all the embeddings corresponding to the tokens
            all_embeds = [e for i, (t, e) in enumerate(zip(sentence.tokens, token_embeddings[model_name][layer]))]
            all_tokens = [t for i, (t, e) in enumerate(zip(sentence.tokens, token_embeddings[model_name][layer]))]
            all_embeds = np.array(all_embeds, dtype=np.float32)
            all_tokens = np.array(all_tokens, dtype=str)
            # print(all_tokens, all_embeds.shape)

            # exclude OOV words' embeddings (they are all NaNs)
            not_nan_tokens = all_tokens[~np.isnan(all_embeds[:, 0])   ]
            not_nan_embeds = all_embeds[~np.isnan(all_embeds[:, 0]), :]

            # make a note of the shape of the vector (n x embed_dim)
            shape = not_nan_embeds.shape
            # TODO: vectorize operation on all tokensnow that it is an numpy array using apply()?
            if filters:
                mask = [all(fn(i, t) for fn_name, fn in filters.items()) for i, t in enumerate(not_nan_tokens)]
            else:
                mask = slice(None, None, None)

            filtered_embeds = not_nan_embeds[mask]
            filtered_shape = filtered_embeds.shape

            # if filtering left no tokens, we will use all_embeds instead
            if filtered_shape[0] == 0:
                io.log(f'filtered embeddings for current sentence are empty. retrying without filters: {sentence.tokens}', type='WARN')

                # now what? use unfiltered (as a fallback)
                filtered_embeds = not_nan_embeds
                
            # [very rarely] if no word has a corresponding embedding, then we have no choice
            # but to return a zero vector (or, sometime in the future, a random vector??)
            if shape[0] == 0:
                filtered_embeds = np.zeros((1, shape[-1]))

            for method in methods:
                # if a pre-defined aggregation method is used, apply it
                if type(method) is str:
                    method_name = method
                    if method_name == 'median':
                        pooled = np.median(filtered_embeds, axis=0).reshape(-1)#.tolist()
                    elif method_name == 'mean':
                        pooled = filtered_embeds.mean(axis=0).reshape(-1) #.tolist()
                    elif method_name == 'last':
                        pooled = filtered_embeds[-1, :].reshape(-1) #.tolist()
                    elif method_name == 'first':
                        pooled = filtered_embeds[0, :].reshape(-1) #.tolist()
                    else:
                        raise ValueError(f'unknown pooling method identifier: {method}')
                # handle the case where a custom aggregation function is applied to the embeddings
                elif type(method) is tuple:
                    method_name, fn = method
                    pooled = fn(filtered_embeds).reshape(-1) #.tolist()

                else:
                    raise ValueError(method)

                model_pooled[method_name][layer] = pooled

        for method_name, layer_wise_reprs in model_pooled.items():
            all_pooled[model_name][method_name] = flatten_activations(layer_wise_reprs)
            all_pooled[model_name][method_name].index = [sentence.uid]

    return all_pooled

