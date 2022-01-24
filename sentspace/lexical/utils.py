import os
import pickle

import numpy as np
import sentspace.utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_mem #, cache_to_disk
from sentspace.utils.misc import merge_lists



# --------- Lexical features

# list of acceptable feature terms to load_databases(...)
# @cache_to_mem
def get_feature_list():
    return ['NRC_Arousal', 'NRC_Valence', 'OSC', 'aoa', 'concreteness', 'lexical_decision_RT',
            'log_contextual_diversity', 'log_lexical_frequency', 'n_orthographic_neighbors', 'num_morpheme',
            'prevalence', 
            'surprisal-3', 'surprisal-1', 'surprisal-2', 'surprisal-4',
            'total_degree_centrality']

def get_feature_list_using_third_party_libraries():
    return ['polysemy', 'num_morpheme_poly']

def get_feature_list_requiring_calculation():
    return ['PMI']


def get_all_features(sentence: 'sentspace.Sentence.Sentence', databases):
    """
    Given list of words, return dict mapping feature to list of feature values
    """
    
    result = {}
    for feature in get_feature_list() + get_feature_list_using_third_party_libraries():
        # we don't want to compute num_morpheme using the dictionary DB by default. 
        # we want to do it only if the polyglot library is unavailable.
        if feature == 'num_morpheme': 
            continue
        computed_feature = get_feature(sentence, feature, databases) 
        # even though we are computing "num_morpheme_poly", we want to
        # record it as "num_morpheme", since the _poly suffix comes from the polyglot library
        # that provides the feature
        if feature == 'num_morpheme_polyglot':
            try:
                import polyglot
            except ImportError:
                feature = 'num_morpheme'
        result[feature] = computed_feature
    return result


def get_feature(sentence: 'sentspace.Sentence.Sentence', feature, databases={}):
    '''
    get specific `feature` for the tokens in `sentence`; fall back to using `lemmas` if necessary
    '''

    def get_feature_(token, lemma, feature):
        """given a `word` and a feature to extract, returns the value of that
            feature for the `word` using available databases

        Args:
            word (str): the token (word) to extract a feature for
            feature (str): name identifier of the feature acc to predefined convention
            databases (dict, in-scope): dictionary of feature --> (word --> feature_value) dictionaries. 
                                        Defaults to {}.

        Returns:
            Any: feature value
        """
        # if the feature is from a database we have on disk
        # database can be a dictionary or an object that implements
        # get(key, default)
        if feature in get_feature_list():
            feature_dict = databases[feature]            
            try:
                return feature_dict[token]
            except KeyError as e:
                try:
                    return feature_dict[lemma]
                except KeyError as e_:
                    return np.nan

        # Other databases we use from libraries we load such as NLTK-Wordnet and Polyglot
        elif feature in get_feature_list_using_third_party_libraries():
            if feature == 'polysemy':
                from nltk.corpus import wordnet
                # first try it with the token itself
                if (synsets := wordnet.synsets(token)):
                    return len(synsets) # TODO does a word's synset include itself?
                # if token is OOV, try again with the lemma
                elif (synsets := wordnet.synsets(lemma)):
                    return len(synsets)
                # otherwise the len of its synset is 1 (itself)
                return 1
            elif feature == 'num_morpheme_poly':
                try:
                    from polyglot.text import Word
                    # try first to obtain # morphemes of the token
                    if (morphed := Word(token, language='en').morphemes):
                        return len(morphed)
                    # otherwise, try using the lemmatized form
                    elif (morphed := Word(lemma, language='en').morphemes):
                        return len(morphed)
                    # if both token and lemma OOV, then return nan? or 1 (i.e. full word is the morpheme?)
                    # but that only means we failed to analyze its morphology, not necessarily that is
                    # *is* a single morpheme
                    return 1 # np.nan
                except ImportError as e:
                    # fall back to simply using a dictionary-based feature
                    # TODO make a note of this somewhere
                    io.log(e.msg, type='WARN')
                    return get_feature_(token, lemma, 'num_morpheme')

        else:
            raise ValueError(f'unable to compute unknown feature `{feature}`')


    features_list = []

    for token, lemma in zip(sentence.tokens, sentence.lemmas):
        features_list += [get_feature_(token, lemma, feature)]

    return features_list


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




@cache_to_mem
def load_databases(features='all', path='.feature_database/', 
                   ignore_case=True,
                  ):
    """
    Load dicts mapping word to feature value
    If one feature, provide in list format
    """
    io.log("loading databases with all features")
    databases = {}
    if features == 'all':
        features = get_feature_list()
    for feature in features:
        if not os.path.exists(path+feature+'.pkl'):
            sentspace.utils.s3.load_feature(key=feature+'.pkl')
        with open(path+feature+'.pkl', 'rb') as f:
            d = pickle.load(f)
            # if ignore_case:  # add lowercase version to feature database
            #     for key, val in d.copy().items():
            #         d[str(key).lower()] = val
            databases[feature] = d

    sanity_check_databases(databases)
    io.log("---done--- loading databases with all features")
    return databases


def sanity_check_databases(databases):
    '''
    perform sanity checks upon loading various datasets
    to ensure validity of the loaded data
    '''

    assert databases['NRC_Arousal']['happy'] == 0.735
    assert databases['NRC_Valence']['happy'] == 1
    assert databases['OSC']['happy'] == 0.951549893181384
    assert abs(databases['aoa']['a'] - 2.893384) < 1e-4
    assert databases['concreteness']['roadsweeper'] == 4.85
    # assert abs(databases['imag']['abbey'] - 5.344) < 1e-4
    assert databases['total_degree_centrality']['a'] == 30
    assert databases['lexical_decision_RT']['a'] == 798.917
    assert abs(databases['log_contextual_diversity']['a'] - 3.9234) < 1e-4
    assert abs(databases['log_lexical_frequency']['a'] - 6.0175) < 1e-4
    assert databases['n_orthographic_neighbors']['a'] == 950.59
    assert databases['num_morpheme']['abbreviated'] == 4
    assert abs(databases['prevalence']['a'] - 1.917) < 1e-3
    assert databases['surprisal-3']['beekeeping'] == 10.258
