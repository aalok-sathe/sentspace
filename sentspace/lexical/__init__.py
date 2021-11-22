import os
from pathlib import Path
from typing import List

import sentspace.utils
from sentspace.lexical import utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk, cache_to_mem


def get_features(sentence: sentspace.Sentence.Sentence, lock=None) -> dict:

    # io.log(f'computing lexical featuures for `{sentence}`')

    # if lock: lock.acquire()
    databases = utils.load_databases(features='all')
    # if lock: lock.release()

    features_from_database = utils.get_all_features(sentence, databases)  # lexical features
    
    # TODO[?] return dict of lists, to be consistent with API?
    # return list of token-level features, as a dict per token
    returnable = []
    for i, token in enumerate(sentence.tokens):
        db_features_slice = {feature: features_from_database[feature][i] for feature in features_from_database}
        returnable += [{'index': sentence.uid,
                        'sentence': str(sentence),
                        'token': token,
                        'lemma': sentence.lemmas[i],
                        'tag': sentence.pos_tags[i],
                        'content_word': sentence.content_words[i],
                        **db_features_slice
                       }]

    return returnable

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
