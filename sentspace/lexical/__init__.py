import os
from pathlib import Path
from typing import List


def get_lexical_features(sentence:List[str]):

    return 
    
    merged_vals = utils.get_all_features_merged(
        flat_cleaned_token_list, flat_lemmatized_token_list, databases)  # lexical features
    # Clear variables so we have RAM
    del databases

    # Results
    result = utils.compile_results(flat_token_list, flat_cleaned_token_list, flat_lemmatized_token_list,
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
