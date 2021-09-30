'''
    ### Sentspace 0.0.1 (C) 2020-2021 EvLab <evlab.mit.edu>, MIT BCS. All rights reserved.

    Homepage: https://github.com/aalok-sathe/sentspace

    Authors (reverse alphabet.; insert `@` symbol for valid email):
    
    - Greta Tuckute `<gretatu % mit.edu>`
    - Aalok Sathe `<asathe % mit.edu>`
    - Alvince Pongos `<apongos % mit.edu>`
    - Josef Affourtit `<jaffourt % mit.edu>`

    Please contact any of the following with questions about the code or license.
    
    - Aalok Sathe `<asathe % mit.edu>` 
    - Greta Tuckute `<gretatu % mit.edu>`

    .. include:: ../README.md
'''

__pdoc__ = {'semantic': False,
            }


from pathlib import Path

import sentspace.utils as utils
import sentspace.syntax as syntax
import sentspace.lexical as lexical
# import sentspace.semantic as semantic
import sentspace.embedding as embedding

from sentspace.Sentence import Sentence
# ...

# TODO: remove processing overhead in API call; ideally the below imports 
# should not exist in this file. TODO: create functions in sentspace.utils
# or otherwise submodule-specific utils to compile such data into a pandas
# dataframe
import pandas as pd
from functools import reduce 
from itertools import chain
from tqdm import tqdm

def run_sentence_features_pipeline(input_file: str, stop_words_file: str = None,
                                   benchmark_file: str = None, output_dir: str = None,
                                   output_format: str = None,
                                   process_lexical: bool = False, process_syntax: bool = False,
                                   process_embedding: bool = False, process_semantic: bool = False,
                                   #
                                   emb_data_dir: str = None) -> Path:
    """
    Runs the full sentence features pipeline on the given input according to
    requested submodules (currently supported: `lexical`, `syntax`, `embedding`,
    indicated by boolean flags).
        
    Returns an instance of `Path` pointing to the output directory resulting from this
    run of the full pipeline. The output directory contains Pickled or TSVed pandas 
    DataFrames containing the requested features.


    Args:
        input_file (str): path to input text file containing sentences
                            one per line [required]
        stop_words_file (str): path to text file containing stopwords to filter
                                out, one per line [optional]
        benchmark_file (str): path to a file containing a benchmark corpus to
                                compare the current input against; e.g. UD [optional]
        
        {lexical,syntax,embedding,semantic,...} (bool): compute submodule features? [False]
    """

    # lock = multiprocessing.Manager().Lock()

    # create output folder
    utils.io.log('creating output folder')
    # (sent_output_path,
    #  glove_words_output_path,
    #  glove_sents_output_path)
    output_dir = utils.io.create_output_paths(input_file,
                                              output_dir=output_dir,
                                              stop_words_file=stop_words_file)
    # with (output_dir / 'config.txt').open('w+') as f:
    #     print(args, file=f)

    utils.io.log('reading input sentences')
    # UIDs, token_lists, sentences = utils.io.read_sentences(input_file, stop_words_file=stop_words_file)
    sentences = utils.io.read_sentences(input_file, stop_words_file=stop_words_file)
    utils.io.log('---done--- reading input sentences')

    # surprisal_database = 'pickle/surprisal-3_dict.pkl' # default: 3-gram surprisal
    # features_ignore_case = True
    # features_transform = ['default', None, None] # format: [method, cols to log transform, cols to z-score] (if method is specified, the other two are ignored)

    # Get morpheme from polyglot library instead of library
    # TODO: where was this supposed to be used?
    # poly_morphemes = utils.get_poly_morpheme(flat_sentence_num, flat_token_list)

    if process_lexical:
        utils.io.log('*** running lexical submodule pipeline')
        _ = lexical.utils.load_databases(features='all')

        # lexical_features = [sentspace.lexical.get_features(sentence, identifier=UIDs[i])
        #                     for i, sentence in enumerate(tqdm(sentences, desc='Lexical pipeline'))]
        lexical_features = utils.parallelize(lexical.get_features, sentences,
                                             wrap_tqdm=True, desc='Lexical pipeline')

        lexical_out = output_dir / 'lexical'
        lexical_out.mkdir(parents=True, exist_ok=True)

        # with (lexical_out/'token-features.json').open('w') as f:
        # 	json.dump(lexical_features, f)

        # lexical is a special case since it returns dicts per token (rather than per sentence)
        # so we want to flatten it so that pandas creates a sensible dataframe from it.
        token_df = pd.DataFrame(chain.from_iterable(lexical_features))

        utils.io.log(f'outputting lexical token dataframe to {lexical_out}')
        if output_format == 'tsv':
            token_df.to_csv(lexical_out / 'token-features.tsv', sep='\t', index=False)
        if output_format == 'pkl':
            token_df.to_pickle(lexical_out / 'token-features.pkl.gz', protocol=5)

        utils.io.log(f'--- finished lexical pipeline')

    if process_syntax:
        utils.io.log('*** running syntax submodule pipeline')
        syntax_features = [syntax.get_features(sentence._raw, dlt=True, left_corner=True, identifier=sentence.uid())
                           for i, sentence in enumerate(tqdm(sentences, desc='Syntax pipeline'))]
        # syntax_features = utils.parallelize(sentspace.syntax.get_features, sentences, UIDs,
        #                                     dlt=True, left_corner=True,
        #                                     wrap_tqdm=True, desc='Syntax pipeline')

        syntax_out = output_dir / 'syntax'
        syntax_out.mkdir(parents=True, exist_ok=True)

        # with (syntax_out/'features.json').open('w') as f:
        # 	f.write(str(syntax_features))

        # put all features in the sentence df except the token-level ones
        token_syntax_features = {'dlt', 'leftcorner'}
        sentence_df = pd.DataFrame([{k: v for k, v in feature_dict.items() if k not in token_syntax_features}
                                    for feature_dict in syntax_features], index=[s.uid() for s in sentences])

        # output gives us dataframes corresponding to each token-level feature. we need to combine these
        # into a single dataframe
        # we use functools.reduce to apply the pd.concat function to all the dataframes and join dataframes
        # that contain different features for the same tokens
        # we use df.T.drop_duplicates().T to remove duplicate columns ('token', 'sentence', 'index' etc) that appear in
        # all/multiple dataframes as part of the standard output schema
        token_dfs = [reduce(lambda x, y: pd.concat([x, y], axis=1, sort=False),
                            (v for k, v in feature_dict.items() if k in token_syntax_features)).T.drop_duplicates().T
                     for feature_dict in syntax_features]
        token_df = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), token_dfs)

        utils.io.log(f'outputting syntax dataframes to {syntax_out}')
        if output_format == 'tsv':
            sentence_df.to_csv(syntax_out / 'sentence-features.tsv', sep='\t', index=False)
            token_df.to_csv(syntax_out / 'token-features.tsv', sep='\t', index=False)
        elif output_format == 'pkl':
            sentence_df.to_pickle(syntax_out / 'sentence-features.pkl.gz', protocol=5)
            token_df.to_pickle(syntax_out / 'token-features.pkl.gz', protocol=5)

        utils.io.log(f'--- finished syntax pipeline')

    # Calculate PMI
    # utils.GrabNGrams(sent_rows,pmi_paths)
    # utils.pPMI(sent_rows, pmi_paths)
    # pdb.set_trace()

    if process_embedding:
        utils.io.log('*** running embedding submodule pipeline')
        # Get GloVE

        stripped_words = utils.text.strip_words(chain(*[s.tokenized() for s in sentences]), method='punctuation')
        vocab = embedding.utils.get_vocab(stripped_words)
        _ = embedding.utils.load_embeddings(emb_file='glove.840B.300d.txt',
                                            vocab=(*sorted(vocab),),
                                            data_dir=emb_data_dir)

        # embedding_features = [sentspace.embedding.get_features(sentence, vocab=vocab, data_dir=emb_data_dir,
        #                                                        identifier=UIDs[i])
        #                        for i, sentence in enumerate(tqdm(sentences, desc='Embedding pipeline'))]
        embedding_features = utils.parallelize(embedding.get_features, 
                                               sentences,
                                            #    [s._raw for s in sentences], [s.uid() for s in sentences],
                                               vocab=vocab, data_dir=emb_data_dir,
                                               wrap_tqdm=True, desc='Embedding pipeline')
        no_content_words = len(sentences)-sum(any(s.content_words()) for s in sentences)
        utils.io.log(f'sentences with no content words: {no_content_words}/{len(sentences)}; {no_content_words/len(sentences):.2f}')

        embedding_out = output_dir / 'embedding'
        embedding_out.mkdir(parents=True, exist_ok=True)

        sentence_df = pd.DataFrame([{k: v for k, v in feature_dict.items() if k != 'token_embeds'}
                                    for feature_dict in embedding_features])

        utils.io.log(f'outputting embedding dataframe(s) to {embedding_out}')
        if output_format == 'tsv':
            sentence_df.to_csv(embedding_out / 'sentence-features.tsv', sep='\t', index=False)
        elif output_format == 'pkl':
            sentence_df.to_pickle(embedding_out / 'sentence-features.pkl.gz', protocol=5)

        # token_dfs = [feature_dict['token_embeds'] for feature_dict in embedding_features]
        # token_df = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), token_dfs)

        # utils.io.log(f'outputting embedding token dataframe to {embedding_out}')
        # token_df.to_csv(embedding_out / 'token-features.tsv', sep='\t', index=False)

        utils.io.log(f'--- finished embedding pipeline')

    # Plot input data to benchmark data
    #utils.plot_usr_input_against_benchmark_dist_plots(df_benchmark, sent_embed)

    if process_semantic:
        pass

    return output_dir
