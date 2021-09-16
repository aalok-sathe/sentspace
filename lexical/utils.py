import os
import pickle

import numpy as np
import sentspace.utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.utils import Word, merge_lists, wordnet


def get_all_features(wordlist, databases):
    """
    Given list of words, return dict mapping feature to list of feature values
    """
    result = {}
    for feature in get_feature_list() + get_feature_list_using_third_party_libraries():
        result[feature] = get_feature(wordlist, feature, databases)
    return result


def get_all_features_merged(flat_token_list, flat_lemmatized_token_list, databases):
    """
    Given list of words & list of lemmatized words,
    return dict mapping feature to list of feature values after merging
    (if a word in its original form exists in the database, use its associated value;
    if not, use value associated with the lemmatized version)
    """
    all_vals = get_all_features(flat_token_list, databases)
    all_vals_lem = get_all_features(flat_lemmatized_token_list, databases)
    merged = {}
    for feature in all_vals:
        merged[feature] = merge_lists(all_vals[feature], all_vals_lem[feature], feature=feature)
    return merged


# @cache_to_mem
def get_is_content(taglst: tuple, content_pos=(wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV)):
    """
    Given list of POS tags, return list of 1 - content word, 0 - not content word
    """
    return tuple(int(text.get_wordnet_pos(tag) in content_pos) for tag in taglst)


# --------- Lexical features

# list of acceptable feature terms to load_databases(...)
# @cache_to_mem
def get_feature_list():
    return ['NRC_Arousal', 'NRC_Valence', 'OSC', 'aoa', 'concreteness', 'lexical_decision_RT',
            'log_contextual_diversity', 'log_lexical_frequency', 'n_orthographic_neighbors', 'num_morpheme',
            'prevalence', 'surprisal-3', 'total_degree_centrality']


def get_feature_list_using_third_party_libraries():
    return ['polysemy', 'num_morpheme_poly']


def get_feature_list_requiring_calculation():
    return ['PMI']


def get_poly_morpheme(sent_num, word_list):
    '''
        Given sent_number and word_list, calculate the morphemes of each word in each sentence
    '''
    raise NotImplementedError




def get_feature(flat_token_list, feature, databases):

    # @cache_to_mem  # (ignore=['databases'])
    def get_feature_(word, feature):
        """given a word and a feature to exatract, returns the value of that
            feature for the word using available databases

        Args:
            word (str): the token (word) to extract a feature for
            feature (str): name identifier of the feature acc to predefined convention
            databases (dict, optional): dictionary of feature --> (word --> feature_value) dictionaries. 
                                        Defaults to {}.

        Returns:
            Any: feature value
        """
        # if the feature is from a database we have on disk
        # database can be a dictionary or an object that implements
        # get(key, default)
        if feature in get_feature_list():
            return databases.get(feature, {}).get(word, np.nan)

        # Other databases we use from libraries we load such as NLTK and Polyglot
        elif feature in get_feature_list_using_third_party_libraries():
            if feature == 'polysemy':
                if wordnet.synsets(word):
                    return len(wordnet.synsets(word))
                return 1
            elif feature == 'num_morpheme_poly':
                morphed = Word(word, language='en').morphemes
                if morphed:
                    return len(morphed)
                return np.nan
        else:
            raise ValueError(f'unable to compute unknown feature `{feature}`')


    d = {}
    features_list = []

    for word in flat_token_list:
        features_list += [get_feature_(word, feature)]

    return features_list


@cache_to_mem
def load_databases(features='all', path='.feature_database/', ignore_case=True):
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
            if ignore_case:  # add lowercase version to feature database
                for key, val in d.copy().items():
                    d[str(key).lower()] = val
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
