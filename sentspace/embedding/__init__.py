import os
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import sentspace.lexical
import sentspace.utils
from pandas.core.frame import DataFrame
from sentspace.embedding import utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from tqdm import tqdm


def get_features(sentence: sentspace.Sentence.Sentence, vocab=None, data_dir=None):
    """get embedding-based features (e.g. avg, min, max, etc.) for sentence.

    Args:
        sentence (str): sentence to get features for
        vocab ([set], optional): vocabulary of all sentences that will be processed in this session.
                                 it is recommended for a calling scope to make this available in order
                                 to save processing time of going through all of Glove each time.
                                 In the future, optimizations may be considered, such as, indexing the
                                 byte offset of a particular token in the Glove file for speedy reading.

    Returns:
        [type]: [description]
    """

    # tokenized = text.tokenize(sentence)
    # tagged_sentence = text.get_pos_tags(tokenized)
    # is_content_word = sentspace.utils.text.get_is_content(tagged_sentence, content_pos=text.pos_for_content)
    # clean words: strip nonletters/punctuation and lowercase
    # nonletters = text.get_nonletters(tokenized, exceptions=[])  # find all non-letter characters in file
    # cleaned_sentence = text.strip_words(tokenized, method='nonletters',
    #                                     nonletters=text.get_nonletters(tokenized, exceptions=[]))
    # lowercased = [*map(lambda x: x.lower(), sentence.tokenized())]

    if vocab is None:
        io.log(f'no vocabulary provided in advance. this may take a while. grab some popcorn ^.^', type='WARN')
    w2v = defaultdict(lambda: defaultdict(None))
    
    # if lock is not None:
    #     lock.acquire()
    w2v['glove'] = utils.load_embeddings(emb_file='glove.840B.300d.txt',
                                         vocab=(*sorted(vocab or sentence.tokenized()),),
                                         data_dir=data_dir)
    # if lock is not None:
    #     lock.release()

    token_embeddings = {
        'glove': utils.get_word_embeds(sentence, w2v=w2v,
                                       which='glove', dims=300),
    }

    content_word_filter = lambda i, token: sentence.content_words()[i]
    filters = {'content_words': content_word_filter}
    pooled_embeddings = utils.pool_sentence_embeds(sentence, token_embeddings, filters=filters)

    lemmatized_sentence = text.get_lemmatized_tokens(sentence.tokenized(), sentence.pos_tagged())
    
    return {
        'index': sentence.uid(),
        'sentence': str(sentence),
        'filters': ','.join(filters.keys()),

        **pooled_embeddings,
    }


    # # Read in benchmark data
    # df_benchmark = pd.read_csv(benchmark_file)

    # # Return percentile per sentence for each
    # percentile_df = utils.return_percentile_df(df_benchmark, sent_embed)
    # print('Writing percentiles')
    # percentile_df.to_csv(bench_perc_out_path, index=False)
