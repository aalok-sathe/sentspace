
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.utils import merge_lists


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


# --------- Lexical features

# list of acceptable feature terms to load_databases(...)
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


@cache_to_disk
def get_feature_(word, feature, databases={}):
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
    if feature in get_feature_list_using_third_party_libraries():
        if feature == 'polysemy':
            if wordnet.synsets(word):
                return len(wordnet.synsets(word))
            return 1
        if feature == 'num_morpheme_poly':
            morphed = Word(x, language='en').morphemes
            if morphed:
                return (len(morphed))
            return np.nan


def get_feature(flat_token_list, feature, databases):
    d = {}
    features_list = []

    for word in flat_token_list:
        features_list += [get_feature_(word, feature, databases)]

    return features_list
