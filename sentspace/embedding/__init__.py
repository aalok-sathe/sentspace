import os
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import pandas as pd
import sentspace.lexical
import sentspace.utils
from pandas.core.frame import DataFrame
from sentspace.embedding import utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from tqdm import tqdm


def get_features(sentence: Union[sentspace.Sentence.Sentence, sentspace.Sentence.SentenceBatch],
                 models_and_methods = [({'glove.840B.300d'}, {'mean', 'median'}),
                                       ({'gpt2-xl', 'distilgpt2', 'gpt2'}, {'last'}),
                                       ({'bert-base-uncased', 'aloxatel/mbert', 'roberta-large'}, {'first'}),
                                      ],
                 vocab=None, data_dir=None):
    """get embedding-based features (e.g. avg, min, max, etc.) for sentence.

    Args:
        sentence (str): sentence to get features for
        models_and_methods (`typing.Collection[typing.Tuple[typing.Collection[str], 
                                                            typing.Collection[typing.Union[str, typing.Callable[np.ndarray]]]]]`):
                this collection maps each embeddings/representation source to the aggregation 
                methods that should be applied to it. None matches all the embeddings/representation sources 
                available. Methods can either be strings specifying standard aggregation methods (first, last, mean, median),
                or a function accepting (n, d) array-like objects and returning (d,) 
                array-like objects where d >= 1.                                 
        vocab (set, optional): vocabulary over all sentences that will be processed in this session.
                                 it is recommended for a calling scope to make this available in order
                                 to save processing time of going through all of Glove each time.
                                 In the future, optimizations may be considered, such as, indexing the
                                 byte offset of a particular token in the Glove file for speedy reading.

    Returns:
        [type]: [description]
    """

    token_embeddings = {}

    if any('glove' in model_name for model_names, _ in models_and_methods for model_name in model_names):
        if vocab is None:
            io.log(f'no vocabulary provided in advance. this may take a while. grab some popcorn ^.^', type='WARN')
        
        for model_names, _ in models_and_methods:
            for model_name in model_names:
                if 'glove' in model_name or 'word2vec' in model_name:
                    w2v = utils.load_embeddings(emb_file=f'{model_name}.txt',
                                                vocab=(*sorted(vocab or sentence.tokens),),
                                                data_dir=data_dir)
                    token_embeddings.update(utils.get_word_embeds(sentence, w2v=w2v, model_name=model_name,
                                                                # TODO: infer dims based on supplied w2v !!
                                                                dims=300))
    for model_names, _ in models_and_methods:
        for model_name in model_names:
            if 'glove' in model_name or 'word2vec' in model_name: continue
            token_embeddings.update(utils.get_huggingface_embeds(sentence, model_name=model_name,
                                                                layers=None)
                                )

    content_word_filter = lambda i, token: sentence.content_words[i]
    filters = {'content_words': content_word_filter}
    
    pooled_embeddings = defaultdict(dict)
    
    for model_names, methods in models_and_methods:
        d = utils.pool_sentence_embeds(sentence, token_embeddings, keys=model_names,
                                       filters=filters, methods=methods)
        # d: model_name -> method_name -> pooled_repr
        for model_name in d:
            pooled_embeddings[model_name].update(d[model_name])

    return {
        'index': sentence.uid,
        'sentence': str(sentence),
        'filters': ','.join(filters.keys()),
        
        'features': pooled_embeddings,
    }


    # # Read in benchmark data
    # df_benchmark = pd.read_csv(benchmark_file)

    # # Return percentile per sentence for each
    # percentile_df = utils.return_percentile_df(df_benchmark, sent_embed)
    # print('Writing percentiles')
    # percentile_df.to_csv(bench_perc_out_path, index=False)
