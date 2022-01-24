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

# __pdoc__ = {'semantic': False,
#             }


from collections import defaultdict
from pathlib import Path

import sentspace.utils as utils
import sentspace.syntax as syntax
import sentspace.lexical as lexical
import sentspace.embedding as embedding

from sentspace.Sentence import Sentence

import pandas as pd
from functools import reduce 
from itertools import chain
from tqdm import tqdm


def run_sentence_features_pipeline(input_file: str, stop_words_file: str = None,
                                   benchmark_file: str = None, output_dir: str = None,
                                   output_format: str = None, batch_size: int = 2_000,
                                   process_lexical: bool = False, process_syntax: bool = False,
                                   process_embedding: bool = False, process_semantic: bool = False,
                                   parallelize: bool = True,
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
    output_dir = utils.io.create_output_paths(input_file,
                                              output_dir=output_dir,
                                              stop_words_file=stop_words_file)
    config_out = (output_dir / 'this_session_log.txt')
    # with config_out.open('a+') as f:
    #     print(args, file=f)

    utils.io.log('reading input sentences')
    sentences = utils.io.read_sentences(input_file, stop_words_file=stop_words_file)
    utils.io.log('---done--- reading input sentences')


    for part, sentence_batch in enumerate(tqdm(utils.io.get_batches(sentences, batch_size=batch_size), 
                                               desc='processing batches')):
        sentence_features_filestem = f'sentence-features_part{part:0>4}'
        token_features_filestem = f'token-features_part{part:0>4}'


        ################################################################################
        #### LEXICAL FEATURES ##########################################################
        ################################################################################
        if process_lexical:
            utils.io.log('*** running lexical submodule pipeline')
            _ = lexical.utils.load_databases(features='all')

            if parallelize:
                lexical_features = utils.parallelize(lexical.get_features, sentence_batch,
                                                    wrap_tqdm=True, desc='Lexical pipeline')
            else:
                lexical_features = [lexical.get_features(sentence)
                                    for _, sentence in enumerate(tqdm(sentence_batch, desc='Lexical pipeline'))]

            lexical_out = output_dir / 'lexical'
            lexical_out.mkdir(parents=True, exist_ok=True)
            utils.io.log(f'outputting lexical token dataframe to {lexical_out}')

            # lexical is a special case since it returns dicts per token (rather than per sentence)
            # so we want to flatten it so that pandas creates a sensible dataframe from it.
            token_df = pd.DataFrame(chain.from_iterable(lexical_features))

            if output_format == 'tsv':
                token_df.to_csv(lexical_out / f'{token_features_filestem}.tsv', sep='\t', index=True)
                token_df.groupby('sentence').mean().to_csv(lexical_out / f'{sentence_features_filestem}.tsv', sep='\t', index=True)
            elif output_format == 'pkl':
                token_df.to_pickle(lexical_out / f'{token_features_filestem}.pkl.gz', protocol=5)
                token_df.groupby('sentence').mean().to_pickle(lexical_out / f'{sentence_features_filestem}.pkl.gz', protocol=5)
            else:
                raise ValueError(f'output format {output_format} not known')

            utils.io.log(f'--- finished lexical pipeline')


        ################################################################################
        #### SYNTAX FEATURES ###########################################################
        ################################################################################
        if process_syntax:
            utils.io.log('*** running syntax submodule pipeline')

            # as an exception, we do *not* parallelize syntax since the backend server is somehow unable to handle
            # multiple requests :(
            syntax_features = [syntax.get_features(sentence._raw, dlt=True, left_corner=True, identifier=sentence.uid)
                                                                        # !!! TODO:DEBUG
                            for i, sentence in enumerate(tqdm(sentences, desc='Syntax pipeline'))]

            syntax_out = output_dir / 'syntax'
            syntax_out.mkdir(parents=True, exist_ok=True)

            # put all features in the sentence df except the token-level ones
            token_syntax_features = {'dlt', 'leftcorner'}
            sentence_df = pd.DataFrame([{k: v for k, v in feature_dict.items() if k not in token_syntax_features}
                                        for feature_dict in syntax_features], index=[s.uid() for s in sentences])

            # output gives us dataframes corresponding to each token-level feature. we need to combine these
            # into a single dataframe
            # we use functools.reduce to apply the pd.concat function to all the dataframes and join dataframes
            # that contain different features for the same tokens
            token_dfs = [reduce(lambda x, y: pd.concat([x, y], axis=1, sort=False),
                                (v for k, v in feature_dict.items() if k in token_syntax_features))
                        for feature_dict in syntax_features]

            # by this point we have merged dataframes with tokens along a column (rather than just a sentence)
            # now we need to stack them on top of each other to have all tokens across all sentences in a single dataframe
            token_df = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), token_dfs)
            token_df = token_df.loc[:, ~token_df.columns.duplicated()]

            utils.io.log(f'outputting syntax dataframes to {syntax_out}')
            if output_format == 'tsv':
                sentence_df.to_csv(syntax_out / f'{sentence_features_filestem}.tsv', sep='\t', index=True)
                token_df.to_csv(syntax_out / f'{token_features_filestem}.tsv', sep='\t', index=False)
            elif output_format == 'pkl':
                sentence_df.to_pickle(syntax_out / f'{sentence_features_filestem}.pkl.gz', protocol=5)
                token_df.to_pickle(syntax_out / f'{token_features_filestem}.pkl.gz', protocol=5)

            utils.io.log(f'--- finished syntax pipeline')

        # Calculate PMI
        # utils.GrabNGrams(sent_rows,pmi_paths)
        # utils.pPMI(sent_rows, pmi_paths)
        # pdb.set_trace()


        ################################################################################
        #### EMBEDDING FEATURES ########################################################
        ################################################################################            
        if process_embedding:
            utils.io.log('*** running embedding submodule pipeline')

            models_and_methods = [
                        # ({'glove.840B.300d'}, {'mean', 'median'}), 
                        # 'distilgpt2',
                        ({'gpt2-xl'}, {'last'}),
                        ({'bert-base-uncased'}, {'first'}),
                     ]
            
            vocab = None
            # does any of the
            if any('glove' in model or 'word2vec' in model for models, _ in models_and_methods for model in models):
                # get a vocabulary across all sentences given as input
                # as the first step, remove any punctuation from the tokens
                stripped_tokens = utils.text.strip_words(chain(*[s.tokens for s in sentences]), method='punctuation')
                # assemble a set of unique tokens
                vocab = set(stripped_tokens)
                # make a spurious function call so that loading glove is cached for subsequent calls
                # TODO allow specifying which glove/w2v version 
                _ = embedding.utils.load_embeddings(emb_file='glove.840B.300d.txt',
                                                    vocab=(*sorted(vocab),),
                                                    data_dir=emb_data_dir)

            if False and parallelize:
                embedding_features = utils.parallelize(embedding.get_features, 
                                                       sentences, models=models, 
                                                       vocab=vocab, data_dir=emb_data_dir,
                                                       wrap_tqdm=True, desc='Embedding pipeline')
            else:
                embedding_features = [embedding.get_features(sentence, models_and_methods=models_and_methods, 
                                                             vocab=vocab, data_dir=emb_data_dir)
                                      for i, sentence in enumerate(tqdm(sentences, desc='Embedding pipeline'))]

            # a misc. stat being computed that needs to be handled better 
            # "no" means no. the stat below is counting how many sentences have NO content words (not to be confused with num. content words)
            no_content_words = len(sentences)-sum(any(s.content_words) for s in sentences)

            utils.io.log(f'sentences without any content words: {no_content_words}/{len(sentences)}; {no_content_words/len(sentences):.2f}')

            embedding_out = output_dir / 'embedding'
            embedding_out.mkdir(parents=True, exist_ok=True)
            
            
            # now we want to output stuff from embedding_features (which is returned by the embedding pipeline)
            # into nicely formatted dataframes.
            # the structure of what is returned by the embedding pipeline is like so:
            #   gpt2-xl:
            #       last: [...] flat multiindexed Pandas series with (layer, dim) as the two indices
            #       mean: 
            #   glove:
            #       mean: [...] flat multiindexed Pandas series with trivially a single layer and 300d, so (1, 300) as the two indices
            # etc.

            # a set of all the models in use
            all_models_methods = {model_name: feature_dict['features'][model_name].keys() 
                                  for feature_dict in embedding_features 
                                    for model_name in feature_dict['features']}

            print(all_models_methods)

            # we want to output BY MODEL
            for model_name in all_models_methods:
                # and BY METHOD
                for method in all_models_methods[model_name]:
                    # each `feature_dict` corresponds to ONE sentence

                    collected = []
                    for feature_dict in embedding_features:

                        # all the keys that contain information such as the sentence, UID, filters used etc,
                        # except for the actual representations obtained from various models.
                        # we need to know this so we can package all this information together with the outputs by model and method
                        metadata_keys = {*feature_dict.keys()} - {'features'} # setminus operator
                        # make a copy of the feature_dict for this sentence excluding the representations themselves
                        meta_df = {key: feature_dict[key] for key in metadata_keys}
                        meta_df.update({'model_name': model_name, 'aggregation': method})
                        meta_df = pd.DataFrame(meta_df, index=[feature_dict['index']])
                        meta_df.columns = pd.MultiIndex.from_product([['metadata'], meta_df.columns, ['']])

                        # model_name -> method -> reprs
                        pooled_reprs = feature_dict['features']
                        flattened_repr = pooled_reprs[model_name][method]

                        collected += [pd.concat([meta_df, flattened_repr], axis=1)]

                    # create further subdirectories by model and aggregation method
                    (embedding_out / model_name / method).mkdir(parents=True, exist_ok=True)

                    sentence_df = pd.concat(collected, axis=0)

                    utils.io.log(f'outputting embedding dataframes for {model_name}-{method} to {embedding_out}')
                    if output_format == 'tsv':
                        sentence_df.to_csv(embedding_out / model_name / method / f'{sentence_features_filestem}.tsv', sep='\t', index=True)
                        # token_df.to_csv(embedding_out / f'{token_features_filestem}.tsv', sep='\t', index=False)
                    elif output_format == 'pkl':
                        sentence_df.to_pickle(embedding_out / model_name / method / f'{sentence_features_filestem}.pkl.gz', protocol=5)
                        # token_df.to_pickle(embedding_out / f'{token_features_filestem}.pkl.gz', protocol=5)


            utils.io.log(f'--- finished embedding pipeline')

        # Plot input data to benchmark data
        #utils.plot_usr_input_against_benchmark_dist_plots(df_benchmark, sent_embed)

        if process_semantic:
            pass


    ################################################################################
    #### \end{run_sentence_features_pipeline} ######################################
    ################################################################################
    return output_dir
