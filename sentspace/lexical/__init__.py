import os
from pathlib import Path
from typing import List
import sentspace.utils
from sentspace.utils import text, io
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.lexical import utils
from sentspace.lexical.content_ratios import get_content_ratio


def get_features(sentence:List[str]):

    lemmatized_sentence = text.get_lemmatized_tokens(sentence, text.get_pos_tags(tuple(sentence)))


    io.log("loading databases with all features")
    databases = io.load_databases(features='all')
    io.log("---done--- loading databases with all features")

    merged_vals = utils.get_all_features_merged(sentence, lemmatized_sentence, databases)  # lexical features
    # Clear variables so we have RAM

    # Results
    result = sentspace.utils.compile_results(flat_token_list, flat_cleaned_token_list, flat_lemmatized_token_list,
                                   flat_pos_tags, flat_is_content_word, setlst,
                                   flat_sentence_num, flat_token_lens, merged_vals)

    result = utils.transform_features(result, *features_transform)

    features_used = ['Age of acquisition', 'Arousal', 'Concreteness',
                    'Contextual diversity (log)', 'Degree centrality (log)',
                    'Frequency of orthographic neighbors (log)', 'Lexical decision RT',
                    'Lexical frequency (log)', 'Lexical surprisal', 'Number of morphemes', 'Number of morphemes poly',
                    'Orthography-Semantics Consistency', 'Prevalence', 'Valence', 'Word length', 'Polysemy']

    print('Computing sentence embeddings')
    sent_embed = utils.get_sent_vectors(result, features_used, embed_method,
                                        content_only=content_only,
                                        pronoun_ratios=pronoun_ratios,
                                        )
    lex_per_word_with_uniform_column = utils.conform_word_lex_df_columns(
        result)
    lex_per_word_with_uniform_column.to_csv(word_lex_output_path, index=False)

    print('Writing lex sentence embedding to csv at ' + embed_lex_output_path)
    sent_embed.to_csv(embed_lex_output_path, index=False)

    # Make the syntax excel sheet
    print('Writing syntax sentence embedding to csv at ' + sent_output_path)
    #pdb.set_trace()

    # Read in benchmark data
    df_benchmark = pd.read_csv(benchmark_file)

    # Return percentile per sentence for each
    percentile_df = utils.return_percentile_df(df_benchmark, sent_embed)
    print('Writing percentiles')
    percentile_df.to_csv(bench_perc_out_path, index=False)
