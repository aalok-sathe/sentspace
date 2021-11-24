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
import pickle 


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

    # Get morpheme from polyglot library instead of library
    # TODO: where was this supposed to be used?
    # poly_morphemes = utils.get_poly_morpheme(flat_sentence_num, flat_token_list)

    if process_lexical:
        utils.io.log('*** running lexical submodule pipeline')
        _ = lexical.utils.load_databases(features='all')

        for i, batch in enumerate(tqdm(utils.io.get_batches(sentences, batch_size=batch_size))):
        
            if parallelize:
                lexical_features = utils.parallelize(lexical.get_features, batch,
                                                     wrap_tqdm=True, desc='Lexical pipeline')
            else:
                lexical_features = [lexical.get_features(sentence)
                                    for _, sentence in enumerate(tqdm(batch, desc='Lexical pipeline'))]

            lexical_out = output_dir / 'lexical'
            lexical_out.mkdir(parents=True, exist_ok=True)
            utils.io.log(f'outputting lexical token dataframe to {lexical_out}')

            # lexical is a special case since it returns dicts per token (rather than per sentence)
            # so we want to flatten it so that pandas creates a sensible dataframe from it.
            token_df = pd.DataFrame(chain.from_iterable(lexical_features))

            if output_format == 'tsv':
                token_df.to_csv(lexical_out / f'token-features_part{i:0>4}.tsv', sep='\t', index=False)
                token_df.groupby('sentence').mean().to_csv(lexical_out / f'sentence-features_part{i:0>4}.tsv', sep='\t', index=False)
            elif output_format == 'pkl':
                token_df.to_pickle(lexical_out / f'token-features_part{i:0>4}.pkl.gz', protocol=5)
                token_df.groupby('sentence').mean().to_pickle(lexical_out / f'sentence-features_part{i:0>4}.pkl.gz', protocol=5)
            else:
                raise ValueError(f'output format {output_format} not known')

        utils.io.log(f'--- finished lexical pipeline')


    if process_syntax:
        utils.io.log('*** running syntax submodule pipeline')

        # as an exception, we do *not* parallelize syntax since the backend server is somehow unable to handle
        # multiple requests :(
        syntax_features = [syntax.get_features(sentence._raw, dlt=True, left_corner=True, identifier=sentence.uid())
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
                                               vocab=vocab, data_dir=emb_data_dir,
                                               wrap_tqdm=True, desc='Embedding pipeline')

        # a misc. stat being computed that needs to be handled better 
        no_content_words = len(sentences)-sum(any(s.content_words()) for s in sentences)

        utils.io.log(f'sentences with no content words: {no_content_words}/{len(sentences)}; {no_content_words/len(sentences):.2f}')

        embedding_out = output_dir / 'embedding'
        embedding_out.mkdir(parents=True, exist_ok=True)

        utils.io.log('temporarily outputting Pickled dictionary of pooled embeddings', type='WARN')
        
        aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for feature_dict in tqdm(embedding_features, desc='embedding output'):
            features = feature_dict['features']
            metadata = {k:feature_dict[k] for k in feature_dict if k != 'features'}
            
            metadata_updated_for_instance = defaultdict(lambda: True)
            for layer in features:
                for method in features[layer]:
                    for which in features[layer][method]:

                        # only update metadata once per model and method; not per layer
                        if f'{which}_{method}' not in metadata_updated_for_instance:
                            for k in metadata: aggregated[which][method][k] += [metadata[k]]
                            metadata_updated_for_instance[f'{which}_{method}']

                        aggregated[which][method][f'layer{layer}'] += [features[layer][method][which]]

                      

        for which in aggregated:
            for method in aggregated[which]:
                out = embedding_out / which / method
                out.mkdir(parents=True, exist_ok=True)

                # X = aggregated[which][method]
                # print(X.keys(), len(X), type(X))
                # for k in X:
                #     print(k, len(X[k]))

                df = pd.DataFrame(aggregated[which][method])
                df.to_pickle(out / f'{which}_{method}.pkl')
                # with (out / f'{which}_{method}.pkl').open('wb') as f:
                #     pickle.dump(aggregated[which][method], f)
                # for layer in feature_dict[method]

        # sentence_df = pd.DataFrame([{k: v for k, v in feature_dict.items() if k != 'token_embeds'}
        #                             for feature_dict in embedding_features])

        # utils.io.log(f'outputting embedding dataframe(s) to {embedding_out}')
        # if output_format == 'tsv':
        #     sentence_df.to_csv(embedding_out / 'sentence-features.tsv', sep='\t', index=False)
        # elif output_format == 'pkl':
        #     sentence_df.to_pickle(embedding_out / 'sentence-features.pkl.gz', protocol=5)

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
