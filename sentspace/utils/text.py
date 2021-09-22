

import string
from collections import Counter

import pandas as pd
import sentspace.utils
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sentspace.utils.caching import cache_to_disk, cache_to_mem

from nltk.tokenize import TreebankWordTokenizer
tokenize = TreebankWordTokenizer().tokenize

pos_for_lemmatization = (wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV) # define POS used for lemmatization
pos_for_content = (wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV)  # define POS that count as content words

# @cache_to_mem
def get_pos_tags(sentence:tuple) -> tuple:
    """
    Given sentence (a list of tokens), return single list of POS tags
    """
    return tuple(tag for token, tag in pos_tag(sentence))


def get_flat_pos_tags(token_lists):
    """
    Given list of sentences (each a list of tokens), return single list of POS tags
    """
    sentspace.utils.io.log('`get_flat_post_tags` '
        'applied to multiple sentences will be DEPRECATED. '
        'please call the appropriate function corresponding to a single sentence.', type='WARN')

    all_pos_tags = []
    for sentence_tokens in token_lists:
        tags = get_pos_tags(sentence_tokens)
        for tag in tags:
            all_pos_tags.append(tag[1])
    return all_pos_tags


def get_flat_sentence_num(token_lists):
    """returns a flattened sentence number list from a list of sentences 

    Args:
        token_lists (List[List[str]]): list of sentences, each of which is a list of tokens

    Returns:
        List[str]: flattened list of sentence number corresponding to each token
    """
    return [1+i for i, sentence in enumerate(token_lists) for token in sentence]


def get_flat_tokens(token_lists):
    '''returns a flattened sentence number list from a list of sentences 

    Args:
        token_lists (List[List[str]]): list of sentences, each of which is a list of tokens

    Returns:
        List[str]: flattened list of tokens
    '''
    return [token for sentence in token_lists for token in sentence]


# def get_sent_num(token_lists):
#     """
#      Given list of sentences (each a list of tokens),
#      return list of sentence no. for each token (starting at 1)
#     """
#     sentence_numbers = []  # sentence number list
#     for i, sentence in enumerate(token_lists):
#         for word in sentence:
#             snlst.append(i+1)
#     return snlst


def get_flat_word_num(token_lists):
    """
    Given list of sentences (each a list of tokens), return list of word no.
    """
    return [1+i for sentence in token_lists for i, token in enumerate(sentence)]


def get_passage_category(lpclst, lplst_word, keyPassageCategory):
    """
    Given category no. for each passage & list of string labels (corresponding to category no.),
    Returns
        1. list of passage category labels for each word
        2. dict mapping category no. to string label
    """
    cat_dict = {num+1: cat for num, cat in enumerate(
        keyPassageCategory)}  # maps category number to category string
    catlst = []
    for passage_num in lplst_word:
        cat_num = lpclst[passage_num-1]
        catlst.append(cat_dict[cat_num])
    return catlst, cat_dict


def get_nonletters(wordlst:tuple, exceptions={'-'}) -> set:
    """
    Given list of tokens, print & return all non-alphabetical characters (except for input exceptions)
    For helping configure how to clean words
    """
    all_chars = set()
    all_nonletters = set()

    for word in wordlst:
        chars, nonletters = get_nonletters_from_word(word, exceptions)
        all_chars.update(chars)
        all_nonletters.update(nonletters)

    # print('All unique characters:', sorted(charlst))
    # print('All non-letter characters in text file:', sorted(nonletters))
    # print('Exceptions passed in:', sorted(exceptions))
    # print('-'*79)
    return all_nonletters

# @cache_to_disk
def get_nonletters_from_word(word, exceptions):
    charlst = set()
    nonletters = set()
    for char in word:
        charlst.add(char)
        if not char.isalpha() and char not in exceptions:
            nonletters.add(char)
    return charlst, nonletters


def strip_words(flat_token_list, method='nonletter', nonletters=None, punctuation=string.punctuation):
    """
    Given list of tokens, return list of cleaned/stripped words. Lower-cased.
    method = 'nonletter': remove all nonletters specified in the nonletters argument
    method = 'punctuation': remove all nonletters specified in the punctuation argument (default value from string.punctuation)
    """
    # if method == 'nonletter':
    #     print('Formatting words - Characters ignored:', nonletters)
    # elif method == 'punctuation':
    #     print('Formatting words - Characters ignored:', punctuation)
    # print('-'*79)
    flat_cleaned_token_list = []
    for t in flat_token_list:
        stripped = strip_word(t, method, nonletters, punctuation)
        if stripped: flat_cleaned_token_list.append(stripped.lower())
    return flat_cleaned_token_list


# @cache_to_disk
def strip_word(word:str, method:str, nonletters:set, punctuation:set) -> str:
    stripped = ''
    for char in word:
        if method == 'nonletter':
            if char not in nonletters:
                stripped += char
        elif method == 'punctuation':
            if char not in punctuation:
                stripped += char
        else:
            raise ValueError(f'unknown method passed to strip word {method}')
    return stripped


def get_token_lens(flat_token_list):
    """get the length of each token in list

    Args:
        flat_tokens_list (List[str]): list of tokens

    Returns:
        List[int]: list of respective lengths
    """    
    return [*map(len, flat_token_list)]


def get_lemmatized_tokens(flat_token_list, flat_pos_tags, lemmatized_pos=[wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV]):
    """
    Given list of tokens & tags, lemmatize nouns & verbs (as specified by input POS tags)
    """
    
    lemmas = WordNetLemmatizer()
    @cache_to_mem
    def lemmatize_token(word, pos):
        if pos in lemmatized_pos:
            return lemmas.lemmatize(word, pos)
        return word

    lemmatized_tokens = []
    for word, POS in zip(flat_token_list, flat_pos_tags):
        pos = get_wordnet_pos(POS)
        lemmatized_tokens.append(lemmatize_token(word, pos))

    return tuple(lemmatized_tokens)

    # n = 0
    # for word, lemma in zip(wordlst, wordlst_lem):
    #     if word != lemma:
    #         n += 1
    # print(
    #     f'Entries for which lemmatized form of word differs from the actual word: {n} words, {n/len(wordlst)*100:.2f}%')
    # print('-'*79)


# @cache_to_mem
def get_is_content(taglst: tuple, content_pos=(wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV)):
    """
    Given list of POS tags, return list of 1 - content word, 0 - not content word
    """
    return tuple(int(get_wordnet_pos(tag) in content_pos) for tag in taglst)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
