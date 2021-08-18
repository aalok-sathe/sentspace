import os
import pickle

import numpy as np
import sentspace.utils
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.utils import Word, merge_lists, wordnet
from sentspace.utils import text


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


def get_vocab(f1g):
    """
    Return set of unique tokens in input (assume input is a list of token lists)
    """
    vocab = set()
    for sent in f1g:
        for token in sent:
            vocab.add(token)
    return vocab


def return_percentile_df(bench_df, usr_df):
    # Initialize df
    perc_df = pd.DataFrame(columns=usr_df.columns)
    # For each sentence get the percentile scores for each feature
    for index, row in usr_df.iterrows():
        temp_df = {}
        # Iterate through the features
        for col in usr_df.columns:
            if col == 'Sentence no.':
                temp_df[col] = row[col]
                continue
            #print(percentileofscore(bench_df[col],row[col]))
            temp_df[col] = percentileofscore(bench_df[col], row[col])
            #pdb.set_trace()
        # Append percentile feature row
        perc_df = perc_df.append(temp_df, ignore_index=True)

    perc_df.drop(columns=['Sentence no.'])
    return perc_df


def read_glove_embed(vocab, glove_path):
    """
    Read through the embedding file to find embeddings for target words in vocab
    Return dict mapping word to embedding (numpy array)
    """
    w2v = {}
    with open(glove_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            w = tokens[0]
            if w in vocab:
                v = tokens[1:]
                w2v[w] = np.array(v, dtype=float)
    return w2v


def get_glove_word(f1g, w2v, return_NA_words=False, save=False, save_path=False):
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
    glove_embed = []
    NA_words = set()
    for i, sent in enumerate(f1g):
        for token in sent:
            if token in w2v:
                glove_embed.append(w2v[token])
            else:
                glove_embed.append([np.nan]*300)
                NA_words.add(token)
    NA_words = list(NA_words)

    flat_token_list = utils.text.get_flat_tokens(f1g)
    flat_sentence_num = utils.text.get_flat_sentence_num(f1g)
    df = pd.DataFrame(glove_embed)
    df.insert(0, 'Sentence no.', flat_sentence_num)
    df.insert(0, 'Word', flat_token_list)

    print(f'Number of words with NA glove embedding: {len(NA_words)},',
          f'{len(NA_words)/len(flat_token_list)*100:.2f}%')
    print('Example NA words:', NA_words[:5])
    print('-'*79)

    if save:
        suffix = save_path.rsplit('.', -1)[1]
        if suffix == 'csv':
            df.to_csv(save_path, index=False)
        elif suffix == 'mat':
            sio.savemat(save_path, {'glove_word': df})
        else:
            raise ValueError('File type not supported!')

    if return_NA_words:
        return df, set(NA_words)
    else:
        return df


def get_glove_sent(df, content_only=False, is_content_lst=None,
                   save=False, save_path=None):
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
    if content_only:
        df = df[np.array(is_content_lst) == 1]
    sent_vectors = df.drop(columns=['Word']).groupby('Sentence no.').mean()  # ignores nans

    na_frac = len(df.dropna())/len(df)
    print(f'Fraction of words used for sentence embeddings: {na_frac*100:.2f}%')
    print('-'*79)

    if save:
        suffix = save_path.rsplit('.', -1)[1]
        if suffix == 'csv':
            sent_vectors.to_csv(save_path)
        elif suffix == 'mat':
            sio.savemat(save_path, {'glove_sent': sent_vectors})
        else:
            raise ValueError('File type not supported!')
    return sent_vectors
