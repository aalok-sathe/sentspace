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


def get_features(sentence:str, identifier=None, vocab=None, data_dir=None):
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

    tokenized = text.tokenize(sentence)
    tagged_sentence = text.get_pos_tags(tokenized)
    is_content_word = sentspace.lexical.utils.get_is_content(tagged_sentence, content_pos=text.pos_for_content)
    # clean words: strip nonletters/punctuation and lowercase
    # nonletters = text.get_nonletters(tokenized, exceptions=[])  # find all non-letter characters in file
    # cleaned_sentence = text.strip_words(tokenized, method='nonletters',
    #                                     nonletters=text.get_nonletters(tokenized, exceptions=[]))
    lowercased = [*map(lambda x: x.lower(), tokenized)]

    if vocab is None:
        io.log(f'no vocabulary provided in advance. this may take a while. grab some popcorn ^.^', type='WARN')
    w2v = defaultdict(lambda: defaultdict(None))
    
    # if lock is not None:
    #     lock.acquire()
    w2v['glove'] = utils.load_embeddings(emb_file='glove.840B.300d.txt',
                                            vocab=(*sorted(vocab or tokenized),),
                                            data_dir=data_dir)
    # if lock is not None:
    #     lock.release()

    token_embeddings = {
        'glove': utils.get_word_embeds(lowercased, w2v=w2v,
                                       which='glove', dims=300),
    }

    # content_word_filter = lambda i, token: is_content_word[i]
    # filters = [content_word_filter]
    pooled_embeddings = utils.pool_sentence_embeds(lowercased, token_embeddings)

    tagged_sentence = text.get_pos_tags(tokenized)
    lemmatized_sentence = text.get_lemmatized_tokens(tokenized, tagged_sentence)
    
    # token_df = pd.DataFrame({'index': identifier,
    #                          'sentence': sentence,
    #                          'tokens': tokenized,
    #                          'lowercased': lowercased,
    #                          'lemmas': lemmatized_sentence,
    #                          'tags': tagged_sentence,
    #                         #  **{k:[e for e in v] for k,v in token_embeddings.items()}
    #                           **token_embeddings,
    #                          }, dtype=str)
    
    # for which in token_embeddings:
    #     # token_df[which] = [v.reshape(-1).tolist() for v in token_embeddings[which]]
    #     token_df[which] = token_embeddings[which]]

    return {
        'index': identifier,
        'sentence': sentence,

        # 'token_embeds': token_df,
        **pooled_embeddings,
    }


    result = sentspace.utils.compile_results(sentence, cleaned_sentence, lemmatized_sentence,
                                             tagged_sentence, content_words, setlst,
                                             flat_sentence_num, flat_token_lens, merged_vals)

    result = utils.transform_features(result, *['default', None, None])

    # print('Computing sentence embeddings')
    # sent_embed = utils.get_sent_vectors(result, features_used, embed_method,
    #                                     content_only=content_only,
    #                                     pronoun_ratios=pronoun_ratios,
    #                                     )
    # lex_per_word_with_uniform_column = utils.conform_word_lex_df_columns(
    #     result)
    # lex_per_word_with_uniform_column.to_csv(word_lex_output_path, index=False)

    # print('Writing lex sentence embedding to csv at ' + embed_lex_output_path)
    # sent_embed.to_csv(embed_lex_output_path, index=False)

    # # Make the syntax excel sheet
    # print('Writing syntax sentence embedding to csv at ' + sent_output_path)
    # #pdb.set_trace()

    # # Read in benchmark data
    # df_benchmark = pd.read_csv(benchmark_file)

    # # Return percentile per sentence for each
    # percentile_df = utils.return_percentile_df(df_benchmark, sent_embed)
    # print('Writing percentiles')
    # percentile_df.to_csv(bench_perc_out_path, index=False)
