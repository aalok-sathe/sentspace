# -*- encoding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from itertools import chain
import string
from collections import defaultdict
import scipy.spatial.distance as ssd
from scipy.stats import zscore, percentileofscore
import os
from datetime import date
import pathlib
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet 
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentspace.s3 import load_feature
import gzip
from polyglot.text import Text, Word

import sys
import pdb

from zs import ZS
from math import log
# from adjustText import adjust_text

lemmas=WordNetLemmatizer()

# --------- General

# divider for print statements
def get_divider():
    return "-"*30
divider = get_divider()

def create_output_path(output_folder, name, analysis, suffix='', sent_suffix=''):
    """
    Return list of file paths and create output folder if appropriate
    Supports analysis = 'lex', 'glove','syntax','PMI'
    """
    if analysis == 'lex':
        return_paths = [f"{name}_lex_features_words{suffix}.csv",
                         f"{name}_lex_features_sents{suffix}{sent_suffix}.csv",
                         f"{name}_plots{suffix}{sent_suffix}_",
                         f"{name}_unique_NA{suffix}{sent_suffix}.csv",
                         f"{name}_benchmark_percentiles{suffix}{sent_suffix}.csv"]
    elif analysis == 'glove':
        return_paths = [f"{name}_glove_words{suffix}.csv",
                        f"{name}_glove_sents{suffix}{sent_suffix}.csv"]
    elif analysis == 'PMI':
        return_paths = [f"{name}_pPMI_0{suffix}.csv",
                    f"{name}_pPMI_1{suffix}.csv",
                    f"{name}_pPMI_2{suffix}.csv",
                    f"{name}_ngrams{suffix}.pkl",
                    f"{name}_nm1grams{suffix}.pkl"]                    
    elif analysis == 'syntax':
        return_paths = [f"{name}_{suffix}.csv",
                        f"{name}_{suffix}{sent_suffix}.csv"]
    elif analysis == 'lex_per_word':
        return_paths = [f"{name}_{suffix}.csv",
                        f"{name}_{suffix}{sent_suffix}.csv"]
    else:
        raise ValueError('Invalid analysis method!')
    output_folder = os.path.join(output_folder, analysis)
    if not os.path.isdir(output_folder): # create output_folder if it doesn't exist
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    result = [os.path.join(output_folder, path) for path in return_paths]
    return result

# --------- Load & preprocess files

def import_sentences(filename, sent_only=False,**kwargs):
    """
    Read txt file of sentences
    If sent_only:
        assumes each line is a sentence
        & return list of sentences as lists of tokens (split by whitespace)
    Else:
        assumes each line is "<passage number>\t<sentence>"
        Returns 1. list of sentences as lists of tokens (split by whitespace)
                2. list of sentences as raw strings
                3. list of passage numbers
    """
    stop_words = kwargs.get('stop_words',None)
    f1g = []
    f1s = [] # strings
    passNum = []

    if sent_only:
        with open(filename, 'r') as file:
            for line in file:
                if stop_words is not None:
                    splitted = line.strip().split()
                    minus_stop_words = [x for x in splitted if x not in stop_words]
                    f1g.append(minus_stop_words)
                    f1.append(' '.join(word[0] for word in minus_stop_words))
                else:
                    f1g.append(line.strip().split())
                    f1s.append(line)

        return f1g, f1s, passNum

    # with open(filename, 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         # line = line[:-1] # remove \n at the end
    #         line = line.split('\t', 1)
    #         passNum.append(line[0])
    #         f1s.append(line[1])
    #         f1g.append(line[1].split())
    # return f1g, f1s, passNum

def import_franklin_sentences_set3(filename):
    f2g = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('passage') or line == '\n':
                pass
            else:
                f2g.append(line.strip('\n').split())
                
    return f2g

def import_data(filename, dtype=None):
    """
    Import text file with \n after each line.
    Pre-computed GloVe vectors can be loaded as:
        glove_embed = import_data('../glove/vectors_243sentences.txt', dtype=lambda x: float(x))

    """
    f1g = []
    with gzip.open(filename+'.zip', 'r') as file:
        for line in file:
            line = line.strip() # remove \n at the end
            tokens = line.split()
            if dtype:
                tokens = [dtype(token) for token in tokens]
            f1g.append(tokens)
    return f1g

def get_wordlst(f1g):
    """
    Given list of sentences (each a list of tokens), return single list of tokens
    """
    wordlst = []
    for sentence in f1g:
        for word in sentence:
            wordlst.append(word)
    return wordlst

def get_pos_tag(f1g):
    """
    Given list of sentences (each a list of tokens), return single list of POS tags
    """
    taglst = []
    for sentence in f1g:
        tags = pos_tag(sentence)
        for tag in tags:
            taglst.append(tag[1])
    return taglst

def get_sent_num(f1g):
    """
     Given list of sentences (each a list of tokens),
     return list of sentence no. for each token (starting at 1)
    """
    snlst = [] #sentence number list
    for i, sentence in enumerate(f1g):
        for word in sentence:
            snlst.append(i+1)
    return snlst

def get_sent_num_passsage(f1g, lplst):
    """
    Given list of passage no. for each sentence, return sentence no. within passage (for each word)
    """
    sent_num = 0
    snplst = []
    current_label = lplst[0]
    for sentence, label in zip(f1g, lplst):
        sent_num += 1
        if label != current_label:
            sent_num = 1
            current_label = label
        for word in sentence:
            snplst.append(sent_num)
    return snplst

def get_word_num(f1g):
    """
    Given list of sentences (each a list of tokens), return list of word no.
    """
    wnslst = []
    for sentence in f1g:
        for i, word in enumerate(sentence):
            wnslst.append(i+1)
    return wnslst

# this might be data-dependent
def load_passage_labels(filename):
    """
    Given .mat file, load and return list of passage no. for each sentence
    """
    labelsPassages = sio.loadmat(filename)
    lP = labelsPassages['labelsPassageForEachSentence']
    return lP.flatten()

# this might be data-dependent
def load_passage_categories(filename):
    """
    Given .mat file, load and return list of passage category labels
    """
    labelsPassages = sio.loadmat(filename)
    lP = labelsPassages['keyPassageCategory']
    return list(np.hstack(lP[0]))

def get_passage_labels(f1g, lplst):
    """
    Given list of passage no. for each sentence, return list of passage no. (for each word)
    """
    lplst_word = []
    for i, sentence in enumerate(f1g):
        for word in sentence:
            lplst_word.append(lplst[i])
    return lplst_word

# this might be data-dependent
def load_passage_category(filename):
    """
    Given .mat file, return category no. for each passage
    """
    labelsPassageCategory = sio.loadmat(filename)
    lPC = labelsPassageCategory['labelsPassageCategory']
    lPC = np.hsplit(lPC,1)
    lpclst = np.array(lPC).tolist()
    lpclst = lpclst[0]
    lpclst = list(chain.from_iterable(lpclst)) # Accessing the nested lists
    return lpclst

def get_passage_category(lpclst, lplst_word, keyPassageCategory):
    """
    Given category no. for each passage & list of string labels (corresponding to category no.),
    Returns
        1. list of passage category labels for each word
        2. dict mapping category no. to string label
    """
    cat_dict = {num+1:cat for num, cat in enumerate(keyPassageCategory)} # maps category number to category string
    catlst = []
    for passage_num in lplst_word:
        cat_num = lpclst[passage_num-1]
        catlst.append(cat_dict[cat_num])
    return catlst, cat_dict

def get_nonletters(wordlst, exceptions=["-"]):
    """
    Given list of tokens, print & return all non-alphabetical characters (except for input exceptions)
    For helping configure how to clean words
    """
    charlst = set()
    nonletters = set()
    for word in wordlst:
        for char in word:
            charlst.add(char)
            if not char.isalpha() and char not in exceptions:
                nonletters.add(char)
    # print('All unique characters:', sorted(charlst))
    print('All non-letter characters in text file:', sorted(nonletters))
    print('Exceptions passed in:', sorted(exceptions))
    print(divider)
    return nonletters

def strip_words(wordlst, method='nonletter', nonletters=None, punctuation=string.punctuation):
    """
    Given list of tokens, return list of cleaned/stripped words. Lower-cased.
    method = 'nonletter': remove all nonletters specified in the nonletters argument
    method = 'punctuation': remove all nonletters specified in the punctuation argument (default value from string.punctuation)
    """
    if method == 'nonletter':
        print('Formatting words - Characters ignored:', nonletters)
    elif method == 'punctuation':
        print('Formatting words - Characters ignored:', punctuation)
    print(divider)
    wordlst_l = []
    for word in wordlst:
        stripped = ""
        for char in word:
            if method == 'nonletter':
                if char not in nonletters:
                    stripped += char
            elif method == 'punctuation':
                if char not in punctuation:
                    stripped += char
            else:
                raise ValueError
        wordlst_l.append(stripped.lower())
    return wordlst_l

def get_wordlen(wordlst_l):
    """
    Return list of word length
    """
    return [len(word) for word in wordlst_l]

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

def get_lemma(wordlst, taglst, lemmatized_pos=[wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV]):
    """
    Given list of tokens & tags, lemmatize nouns & verbs (as specified by input POS tags)
    """
    wordlst_lem = []
    for word, POS in zip(wordlst, taglst):
        pos = get_wordnet_pos(POS)
        if pos in lemmatized_pos:
            wordlst_lem.append(lemmas.lemmatize(word, pos))
        else:
            wordlst_lem.append(word)

    n = 0
    for word, lemma in zip(wordlst, wordlst_lem):
        if word != lemma:
            n += 1
    print(f'Entries for which lemmatized form of word differs from the actual word: {n} words, {n/len(wordlst)*100:.2f}%')
    print(divider)

    return wordlst_lem

def get_is_content(taglst, content_pos=[wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV]):
    """
    Given list of POS tags, return list of 1 - content word, 0 - not content word
    """
    is_content_lst = []
    for tag in taglst:
        if get_wordnet_pos(tag) in content_pos:
            is_content_lst.append(1)
        else:
            is_content_lst.append(0)
    print("All POS in text:", sorted(set(taglst)))
    print("Content words defined as:", sorted(content_pos))
    print(f"Number of content words: {sum(is_content_lst)}, {sum(is_content_lst)/len(taglst)*100:.2f}%")
    print(divider)
    return is_content_lst

# --------- Lexical features

def load_surprisal(file='pickle/surprisal-3_dict.pkl'):
    """
    Load dict mapping word to surprisal value
    """
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_databases(features='all', path='.feature_database/', ignore_case=True):
    """
    Load dicts mapping word to feature value
    If one feature, provide in list format
    """
    databases = {}
    if features == 'all':
        features = get_feature_list()
    for feature in features:
        if not os.path.exists(path+feature+'.pkl'):
            load_feature(key=feature+'.pkl')
        with open(path+feature+'.pkl', 'rb') as f:
            d = pickle.load(f)
            if ignore_case: # add lowercase version to feature database
                for key, val in d.copy().items():
                    d[str(key).lower()] = val
            databases[feature] = d
    return databases

# list of acceptable feature terms to load_databases()
def get_feature_list():
    return ['NRC_Arousal', 'NRC_Valence', 'OSC', 'aoa', 'concreteness', 'lexical_decision_RT', 
    'log_contextual_diversity', 'log_lexical_frequency', 'n_orthographic_neighbors', 'num_morpheme', 
    'prevalence', 'surprisal-3', 'total_degree_centrality']

def get_feature_list_using_third_party_libraries():
    return ['polysemy','num_morpheme_poly']

def get_feature_list_requiring_calculation():
    return ['PMI']

def get_poly_morpheme(sent_num, word_list):
    '''
        Given sent_number and word_list, calculate the morphemes of each word in each sentence
    '''



def get_pronoun_ratio(sent_num, tag_list):
    """
    Given sentence number and parts of speech tag corresponding to this sentence's word, calculate the pronoun ratio
    """
    df = pd.DataFrame({'sent_num':sent_num,'tag_list':tag_list})

    # initialize pronoun tags
    pronoun_tags = ['PRP','PRP$','WP','WP$']

    # Per sentence get the ratio
    p_ratios = {}
    for sent in df.sent_num.unique():
        t_df = df[df.sent_num == sent]['tag_list']
        counts = t_df.value_counts()

        p_count = 0
        not_p_count = 0
        for tag in counts.keys():
            if tag in pronoun_tags:
                p_count += counts[tag]
            else:
                not_p_count += counts[tag]
        p_ratios[sent] = p_count/(p_count + not_p_count)#f'{p_count}:{not_p_count}'
    df_p_ratio = pd.DataFrame({'sent_num':p_ratios.keys(),'pronoun_ratio':p_ratios.values()})
    return df_p_ratio

def get_content_ratio(sent_num, tag_list):
    """
    Given sentence number and boolean tag corresponding to this sentence's word, calculate the content ratio
    """
    df = pd.DataFrame({'sent_num':sent_num,'tag_list':tag_list})

    # Per sentence get the ratio
    cont_ratios = {}
    for sent in df.sent_num.unique():
        # Grab just the specific sentences
        t_df = df[df.sent_num == sent]['tag_list']

        # Get the ratio of content to total 
        cont_ratios[sent] = t_df.sum()/len(t_df)

    df_cont_ratio = pd.DataFrame({'sent_num':cont_ratios.keys(),'content_ratio':cont_ratios.values()})
    return df_cont_ratio

def getFeature(wordlist, feature, databases):
    """
    Given list of words, return list of corresponding feature values
    """
    d = {}
    feature_lst = []

    # Some databases we create and use
    if feature in get_feature_list():
        d = databases[feature]
        for word in wordlist:
            if word in d:
                feature_lst.append(d[word])
            else:
                feature_lst.append(np.nan)

    # Other databases we use from libraries we load such as NLTK and Polyglot
    elif feature in get_feature_list_using_third_party_libraries():
        if feature == 'polysemy':
            for x in wordlist:
                if wn.synsets(x):
                    feature_lst.append( len(wn.synsets(x)) )
                else:
                    feature_lst.append(1)
        if feature == 'num_morpheme_poly':
            for x in wordlist:
                morphed = Word(x, language='en').morphemes
                if morphed:
                    feature_lst.append(len(morphed))
                else:
                    feature_lst.append(np.nan)

    return feature_lst

def mergeList(nonLemVals,lemVals, feature=""):
    '''Input: Two lists.
       Return: If list 1 contains NA vals, the NA val is replaced by the value in list 2 (either numerical val or np.nan again)
    '''
    count_lem = 0
    count_na = 0
    count_nonlem = 0
    mergeLst = []
    for val1, val2 in zip(nonLemVals, lemVals):
        if not np.isnan(val1):
            mergeLst.append(val1)
            count_nonlem += 1
        else:
            mergeLst.append(val2)
            if not np.isnan(val2):
                count_lem += 1
            else:
                count_na += 1
    n = len(mergeLst)
    print(feature, f"| number of values derived from original form: {count_nonlem}, {count_nonlem/n*100:.2f}%")
    print(feature, f"| number of values derived from lemmatized form: {count_lem}, {count_lem/n*100:.2f}%")
    print(feature, f"| number of values = NA: {count_na}, {count_na/n*100:.2f}%")
    print(divider)
    return mergeLst

def get_all_features(wordlist, databases):
    """
    Given list of words, return dict mapping feature to list of feature values
    """
    result = {}
    for feature in get_feature_list():
        result[feature] = getFeature(wordlist, feature, databases)
    for feature in get_feature_list_using_third_party_libraries():
        result[feature] = getFeature(wordlist, feature, databases)
    return result

def get_all_features_merged(wordlst, wordlst_lem, databases):
    """
    Given list of words & list of lemmatized words,
    return dict mapping feature to list of feature values after merging
    (if a word in its original form exists in the database, use its associated value;
    if not, use value associated with the lemmatized version)
    """
    all_vals = get_all_features(wordlst, databases)
    all_vals_lem = get_all_features(wordlst_lem, databases)
    merged = {}
    for feature in all_vals:
        merged[feature] = mergeList(all_vals[feature], all_vals_lem[feature], feature=feature)
    return merged

# def compile_results(wordlst, wordlst_l, wordlst_lem, taglst, is_content_lst, setlst,
#                     snlst, lplst_word, snplst, wnslst, catlst, wordlen, merged_vals):
def compile_results(wordlst, wordlst_l, wordlst_lem, 
                    taglst, is_content_lst, setlst,
                    snlst, wordlen, merged_vals):
    """
    Return dataframe: each row is a word & its various associated values
    """
    result = pd.DataFrame({'Word': wordlst})
    result['Word cleaned'] = wordlst_l
    result['Word lemma'] = wordlst_lem

    result['POS'] = taglst
    result['Content/function'] = is_content_lst
    result['Set no.'] = setlst
    result['Sentence no.'] = snlst
    #result['Passage no.'] = lplst_word
    #result['Sentence no. within passage'] = snplst
    #result['Word no. within sentence'] = wnslst
    #result['Broad topic'] = catlst
    result['Specific topic'] = ['']*len(wordlst)
    result['Word length'] = wordlen
    result['polysemy'] = merged_vals['polysemy']

    # List what you want the columns to be called
    cols = {'NRC_Arousal': 'Arousal', 
            'NRC_Valence': 'Valence', 
            'OSC': 'Orthography-Semantics Consistency', 
            'aoa': 'Age of acquisition', 
            'concreteness': 'Concreteness',
            'lexical_decision_RT': 'Lexical decision RT',
            'log_contextual_diversity': 'Contextual diversity (log)',
            'log_lexical_frequency': 'Lexical frequency (log)',
            'n_orthographic_neighbors': 'Frequency of orthographic neighbors',
            'num_morpheme': 'Number of morphemes',
            'prevalence': 'Prevalence',
            'surprisal-3': 'Lexical surprisal',
            'total_degree_centrality': 'Degree centrality',
            'polysemy':'Polysemy',
            'num_morpheme_poly':'Number of morphemes poly',
            #'Pronoun Ratio':'Pronoun Ratio'
    }

    for key, val in cols.items():
        result[val] = merged_vals[key]
    return result
    
def compile_results_for_glove_only(wordlst, wordlst_l, wordlst_lem, 
                    taglst, is_content_lst, setlst,
                    snlst, wordlen):
    """
    Return dataframe: each row is a word & its various associated values
    """
    result = pd.DataFrame({'Word': wordlst})
    result['Word cleaned'] = wordlst_l
    result['Word lemma'] = wordlst_lem

    result['POS'] = taglst
    result['Content/function'] = is_content_lst
    result['Set no.'] = setlst
    result['Sentence no.'] = snlst
    result['Specific topic'] = ['']*len(wordlst)
    result['Word length'] = wordlen
    return result
    
def conform_word_lex_df_columns(df):
    # List what you want the columns to be called
    cols = {'NRC_Arousal': 'Arousal', 
            'NRC_Valence': 'Valence', 
            'OSC': 'Orthography-Semantics Consistency', 
            'aoa': 'Age of acquisition', 
            'concreteness': 'Concreteness',
            'lexical_decision_RT': 'Lexical decision RT',
            'log_contextual_diversity': 'Contextual diversity (log)',
            'log_lexical_frequency': 'Lexical frequency (log)',
            'n_orthographic_neighbors': 'Frequency of orthographic neighbors',
            'num_morpheme': 'Number of morphemes',
            'prevalence': 'Prevalence',
            'surprisal-3': 'Lexical surprisal',
            'total_degree_centrality': 'Degree centrality',
            'polysemy':'Polysemy',
            'num_morpheme_poly':'Number of morphemes poly',
    }
    df.rename(columns=cols)

    # Remove empty column that are vestiges of temporary analyses
    df = df.drop(columns=['Specific topic'])
    return df

def transform_features(df, method='default', cols_log=None, cols_z=None):
    df = df.copy()
    if method == 'default':
        cols_log = ['Degree centrality', 'Frequency of orthographic neighbors']
    if cols_log:
        for col in cols_log:
            df[col] = np.log10(df[col].astype('float')+1)
            df = df.rename({col: col+' (log)'})
        df = df.rename(columns={col: col+' (log)' for col in cols_log})
    if cols_z:
        for col in cols_z:
            df[col] = zscore(df[col].astype('float'), nan_policy='omit')
        df = df.rename(columns={col: col+' (z)' for col in cols_z})
    return df
    
#     return df_main

def countNA(lst):
    """
    Return number of NAs in a list
    """
    return sum(np.isnan(lst))

def countNA_df(df, features='all'):
    """
    Given dataframe of words and feature values
    Return list of number of NAs in each word's features
    """
    if features == 'all':
        features = ['Age of acquisition', 'Concreteness', 'Prevalence', 'Arousal', 'Valence', 'Dominance', 'Ambiguity: percentage of dominant', 'Log lexical frequency', 'Lexical surprisal', 'Word length']
    df = df[features]
    return list(df.isnull().sum(axis=1))

def uniqueNA(df, feature):
    """
    Given dataframe of words and feature values & desired feature,
    return set of unique words with NA in given feature
    """
    return sorted(set(df['Word cleaned'][df[feature].isna()]))

def avgNA(result, feature):
    """
    Return fractions of words with NA (for given feature) in each sentence
    """
    return result.groupby('Sentence no.').apply(lambda data: countNA(data[feature])/len(data))

def get_NA_words(result, wordlst_l, features):
    """
    Return list of words that have NA in at least one of the specified features
    """
    big_u_lst = []
    for feature in features:
        big_u_lst.extend(uniqueNA(result, feature))
    u_lst = sorted(set(big_u_lst))
    return (big_u_lst, u_lst)

def plot_bar(labels_plot, func, ylabel, title, yerr_func=None):
    """
    Return bar plot & list of values plotted
    given function to apply for each feature
    """
    x = np.arange(len(labels_plot))
    y = [func(label) for label in labels_plot]
    fig, ax = plt.subplots()
    plt.grid(color='grey', which='both',linestyle='-', axis='y',linewidth=0.5)
    if yerr_func: # error bars
        yerr = [yerr_func(label) for label in labels_plot]
        ax.bar(x, y, yerr=yerr, align='center', alpha=0.5)
    else:
        ax.bar(x, y, align='center', alpha=0.5)
    ax.set(ylabel=ylabel, title=title)
    plt.xticks(x, labels_plot, rotation='vertical');
    return fig, y

def plot_all(result, labels_plot, wordlst_l, save=False, save_path=None):
    """
    Bar plots on NAs in data
    """
    numSentences = len(np.unique(result['Sentence no.']))
    numUniqueWords = len(np.unique(result['Word cleaned']))

    figA, yA = plot_bar(labels_plot, lambda feature: countNA(result[feature]),
             ylabel=f'No. NA values out of {len(result)} words',
             title='A. No. NA values for each feature');
    figB, yB = plot_bar(labels_plot, lambda feature: len(uniqueNA(result, feature)),
             ylabel=f'No. unique NA values out of {len(result)} words ({numUniqueWords} unique words)',
             title='B. No. unique words with NA values for each feature');
    figC, yC = plot_bar(labels_plot, lambda feature: np.mean(avgNA(result, feature)),
         ylabel='Mean proportion of NA values within each sentence',
         title='C. Mean proportion of words with NA values within each sentence', #  \n Errorbars denote std across sentences'
                 yerr_func = lambda feature: np.std(avgNA(result, feature)));

    def findProp(v, cutoff=0.5):
        return sum(v >= cutoff)

    figD, yD = plot_bar(labels_plot, lambda feature: findProp(avgNA(result, feature)),
             ylabel=f'No. sentences out of {numSentences} sentences',
             title='D. No. sentences with mean proportion of NA values .50 or higher');
    if save:
        for fig, name in zip([figA, figB, figC, figD], ['A', 'B', 'C', 'D']):
            fig.savefig(save_path + name + '.png', bbox_inches = 'tight')
    return figA, figB, figC, figD

def avg_feature(data, feature, method):
    """
    Return average value of feature
    """
    if method=='strict':
        data = data.dropna()
    elif method=='all':
        pass
    else:
        raise ValueError('Method not recognized')
    return np.nanmean(np.array(data[feature], dtype=float))

def get_sent_vectors(df, features, method='strict', content_only=False,
                     save=False, save_path=None, **kwargs):
    """
    Return dataframe of sentence embeddings (each row as a sentence)
    Method:
        'strict' - if a word has NA in any feature, it is skipped in the sentence average for all features
        'all' - use all non-NA values for sentence average in any feature
    content_only - if True, use content words only in a sentence
    """
    pronoun_ratios = kwargs.get('pronoun_ratios', None)
    content_ratios = kwargs.get('content_ratios', None)
    
    if content_only:
        df = df[df["Content/function"] == 1]
    sent_vectors = pd.DataFrame({'Sentence no.': df['Sentence no.'].unique()})
    df = df[features + ['Sentence no.']].groupby('Sentence no.')
    for name, feature in zip(features, features):
        sent_vectors[name] = list(df.apply(lambda data: avg_feature(data, feature, method)))
    if pronoun_ratios is not None:
        sent_vectors['Pronoun ratios'] = pronoun_ratios['pronoun_ratio']
    # if content_ratios is not None:
    #     sent_vectors['Content ratios'] = content_ratios['content_ratio']
    if save:
        sio.savemat(save_path, {'sent_vectors': sent_vectors.drop(columns=['Sentence no.']).to_numpy()})
    return sent_vectors

def get_differential_sents(embed1, embed2, n, result, method='euclidean'):
    """
    Print sentences with the largest distance between the two input embeddings
    Return index of these sentences (return 1-indexed; assume sentence no. are 1-indexed)
    """
    if method == 'euclidean':
        func = ssd.euclidean
    elif method == 'correlation':
        func = ssd.correlation
    elif method == 'cosine':
        func = ssd.cosine
    else:
        raise ValueError('Method not implemented')
    diff = np.array([func(embed1[i], embed2[i]) for i in range(len(embed1))])
    top_diff_ind = (-diff).argsort(axis=None)[:n]
    top_diff_sent_no = [i+1 for i in top_diff_ind]
    print('Sentences with largest differences:', top_diff_sent_no)
    for i, idx in enumerate(top_diff_ind):
        sent_no = idx+1
        sent = result[result['Sentence no.'] == sent_no].sort_values('Word no. within sentence')
        print(f'{i+1}, sentence {sent_no}: ', list(sent['Word']))
        # print('Number of NA features for a word:', countNA_df(sent, features='all'))
        # print(f'Value in embedding 1: {x[idx]}, embedding 2: {y[idx]}')
        print(f'Distance: {diff[idx]}')
        print()
    return top_diff_sent_no

def annotateDecomp(X_decomp, annotation, adjustment_method = 'skip', x_min = -0.02, x_max = 0.02, y_min = -0.01, y_max = 0.02, skip_no = 2, dims=[0,1], save=False):
    fig, ax = plt.subplots(figsize=(35,25))
    # plot all points
    plt.plot(X_decomp[:, dims[0]], X_decomp[:, dims[1]], 'bo')

    plotted_texts = []
    added_indices = [1]
    
    if adjustment_method == 'simple':
        texts = [plt.text(X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i], ha='center', va='center') for i in range(len(X_decomp[:,0]))]
        adjust_text(texts)
        
    if adjustment_method == 'none':
        texts = [plt.text(X_decomp[i, dims[0]] * (1 + 0.01), X_decomp[i, dims[1]] * (1 + 0.02), annotation[i], ha='center', va='center') for i in range(len(X_decomp[:,0]))]
    
    if adjustment_method == 'skip':
        for i in range(len(X_decomp[:,0])):
            if x_min < X_decomp[i,0] < x_max:
                if y_min < X_decomp[i,1] < y_max:
                    if i % skip_no == 0:
                        texts = plt.text(X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i], ha='center', va='center')
                        plotted_texts.append(texts)
            else: 
                texts = plt.text(X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i], ha='center', va='center')
                plotted_texts.append(texts)
        adjust_text(plotted_texts)

    if adjustment_method == 'distance': # OBS HEAVY
        x_dists = []
        for k in added_indices:
            x_dist = np.abs(X_decomp[i,0] - X_decomp[k,0])
            x_dists.append(x_dist)
            if min(x_dists) > frac:
                y_dists = []
                for g in added_indices:
                    y_dist = np.abs(X_decomp[i,1] - X_decomp[g,1])
                    y_dists.append(y_dist)
                    if min(y_dists) > frac:
                        # plot
                        texts = plt.text(X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i], ha='center', va='center')
                        plotted_texts.append(texts)
                        added_indices.append(i)
        adjust_text(plotted_texts)
        
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()    
        
    if save:
        fig.savefig(save, dpi=240)
   

# --------- GloVe
def lowercase(f1g):
    """
    Return lowercase version of input (assume input is a list of token lists)
    """
    return [[token.lower() for token in sent] for sent in f1g]

def get_sent_version(version, df):
    """
    Return a list of sentences as lists of tokens given dataframe & version of token to use
    Options for version: 'raw', 'cleaned', 'lemmatized'
    """
    ref = {'raw': 'Word', 'cleaned': 'Word cleaned', 'lemmatized': 'Word lemma'}
    version = ref[version]
    f1g = []
    for i in df['Sentence no.'].unique():
        f1g.append(list(df[df['Sentence no.'] == i].sort_values('Word no. within sentence')[version]))
    return f1g

def get_vocab(f1g):
    """
    Return set of unique tokens in input (assume input is a list of token lists)
    """
    vocab = set()
    for sent in f1g:
        for token in sent:
            vocab.add(token)
    return vocab

def read_glove_embed(vocab, glove_path):
    """
    Read through the embedding file to find embeddings for target words in vocab
    Return dict mapping word to embedding (numpy array)
    """
    w2v = {}
    with open(glove_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            w = tokens[0]
            if w in vocab:
                v = tokens[1:]
                w2v[w] = np.array(v, dtype=float)
    return w2v

def get_glove_word(f1g, w2v, return_NA_words=False, save=False, save_path=False):
    """
    Return dataframe of each word, sentence no., and its glove embedding
    If embedding does not exist for a word, fill cells with np.nan
    Parameters:
        f1g: list of sentences as lists of tokens
        w2v: dict mapping word to embedding
        return_NA_words: optionally return unique words that are NA
        save: whether to save results
        save_path: path to save, support .csv & .mat files
    """
    glove_embed = []
    NA_words = set()
    for i, sent in enumerate(f1g):
        for token in sent:
            if token in w2v:
                glove_embed.append(w2v[token])
            else:
                glove_embed.append([np.nan]*300)
                NA_words.add(token)
    NA_words = list(NA_words)

    wordlst = get_wordlst(f1g)
    snlst = get_sent_num(f1g)
    df = pd.DataFrame(glove_embed)
    df.insert(0, 'Sentence no.', snlst)
    df.insert(0, 'Word', wordlst)

    print(f'Number of words with NA glove embedding: {len(NA_words)},',
          f'{len(NA_words)/len(wordlst)*100:.2f}%')
    print('Example NA words:', NA_words[:5])
    print(divider)

    if save:
        suffix = save_path.rsplit('.', -1)[1]
        if suffix == 'csv':
            df.to_csv(save_path, index=False)
        elif suffix == 'mat':
            sio.savemat(save_path, {'glove_word': df})
        else:
            raise ValueError('File type not supported!')

    if return_NA_words:
        return df, set(NA_words)
    else:
        return df

def get_glove_sent(df, content_only=False, is_content_lst=None,
                    save=False, save_path=None):
    """
    Return dataframe of each sentence no. and its sentence embedding
    from averaging embeddings of words in a sentence (ignore NAs)
    Parameters:
        df: dataframe, output of get_glove_word()
        content_only: if True, use content words only
        is_content_lst: list, values 1 if token is content word, 0 otherwise
        save: whether to save results
        save_path: path to save, support .csv & .mat files
    """
    if content_only:
        df = df[np.array(is_content_lst) == 1]
    sent_vectors = df.drop(columns=['Word']).groupby('Sentence no.').mean() # ignores nans

    na_frac = len(df.dropna())/len(df)
    print(f'Fraction of words used for sentence embeddings: {na_frac*100:.2f}%')
    print(divider)

    if save:
        suffix = save_path.rsplit('.', -1)[1]
        if suffix == 'csv':
            sent_vectors.to_csv(save_path)
        elif suffix == 'mat':
            sio.savemat(save_path, {'glove_sent': sent_vectors})
        else:
            raise ValueError('File type not supported!')
    return sent_vectors

def return_percentile_df(bench_df, usr_df):
    # Initialize df
    perc_df = pd.DataFrame(columns = usr_df.columns)
    # For each sentence get the percentile scores for each feature
    for index, row in usr_df.iterrows():
        temp_df = {}
        # Iterate through the features
        for col in usr_df.columns:
            if col == 'Sentence no.':
                temp_df[col] = row[col]
                continue
            #print(percentileofscore(bench_df[col],row[col]))
            temp_df[col] = percentileofscore(bench_df[col],row[col])
            #pdb.set_trace()
        # Append percentile feature row
        perc_df = perc_df.append(temp_df,ignore_index=True)

    perc_df.drop(columns=['Sentence no.'])
    return perc_df

def plot_usr_input_against_benchmark_dist_plots(bench_df, usr_df):
    '''
    This function plots the lexical feature data. So it accepts a df of the form that gets saved at the very end of the pipeline.
    bench_df: A df that Greta et al determined would be a benchmark (i.e. a corpus that we passed through the pipeline)
    usr_df: This is the df that the user would like to compare against the benchmark
    '''
    plt.figure(figsize=(17,10))
    plt.rcParams.update({'font.size': 5})
    plot_number = 1
    binz = 13
    for col in usr_df.columns:
        if 'Sentence no' in col:
            #plot_number = plot_number + 1
            continue
        ax = plt.subplot(4, 5, plot_number)
        bench_df[col].hist(bins=binz, density=True, alpha=0.25,ax=ax, color="skyblue")
        bench_df[col].plot.kde(color="skyblue")
        usr_df[col].hist(bins=binz, density=True, alpha=0.25,ax=ax, color="orange")
        usr_df[col].plot.kde(color="orange",title=col)

        # plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['benchmark','usr_input'])

        # Go to the next plot for the next loop
        plot_number = plot_number + 1

    plt.show()

########## PMI Block ##############
def GrabNGrams(sentences, save_paths):
    '''
    save paths is a list = 
        ['output_folder/03252021/PMI/example_pPMI_0.csv', 
        'output_folder/03252021/PMI/example_pPMI_1.csv', 
        'output_folder/03252021/PMI/example_pPMI_2.csv', 
        'output_folder/03252021/PMI/example_ngrams.pkl', 
        'output_folder/03252021/PMI/example_nm1grams.pkl']
    '''
    sample = sentences
    
    google1 = ZS('PMI/google-books-eng-us-all-20120701-1gram.zs')
    google2 = ZS('PMI/google-books-eng-us-all-20120701-2gram.zs')

    #  break sentences into strings
    def populate(sentences):
        ngra = dict()
        nm1gra = dict()
        for sentence in sentences:
            tokens = sentence.lower().split()
            tokens = ['_START_'] + tokens + ['_END_']
            for t in range(0, len(tokens) - 1):
                ngra[(tokens[t], tokens[t + 1])] = 0
                #print 0, (tokens[t], tokens[t + 1])
                nm1gra[tokens[t]] = 0
            for t in range(0, len(tokens) - 2):
                ngra[(tokens[t], tokens[t + 2])] = 0
                #print 1, (tokens[t], tokens[t + 2])
            for t in range(0, len(tokens) - 3):
                ngra[(tokens[t], tokens[t + 3])] = 0
                #print 2, (tokens[t], tokens[t + 3])
            nm1gra[tokens[len(tokens) - 1]] = 0
        for t1, t2 in ngra.copy().keys():
            ngra[(t2, t1)] = 0
        return ngra, nm1gra
    
    ngrams, nm1grams = populate(sample)
    
    #  fetch ngram and n-1gram
    def fetch(ngra, z=google2, zm1=google1):
        ngram_c = 0
        ngram_str = " ".join(ngra)
        #pdb.set_trace()
        for record in z.search(prefix=ngram_str):
            entry = record.split()
            if entry[1] == ngra[1]:
                ngram_c += int(entry[3])
        if nm1grams[ngra[0]] > 0:
            nm1gram_c = nm1grams[ngra[0]]
        else:
            nm1gram_c = 0
            for record in zm1.search(prefix=ngra[0]):
                entry = record.split()
                if entry[0] == ngra[0]:
                    nm1gram_c += int(entry[2])
        return ngram_c, nm1gram_c
    
    surprisals = dict()
    for ngram in ngrams.copy().keys():
        #print ngram
        #pdb.set_trace()
        ngrams[ngram], nm1grams[ngram[0]] = fetch(ngram)
    
    
    #with open(save_path+'/PMI/ngrams.pkl', 'w') as f:
    with open(save_paths[3], 'w') as f:
        pdb.set_trace()
        pickle.dump(ngrams, f)
    
    
    #with open('PMI/nm1grams.pkl', 'w') as f:
    with open(save_paths[4], 'w') as f:
        pickle.dump(nm1grams, f)


def pPMI(sentences, save_paths):
    '''
    save paths is a list = 
        ['output_folder/03252021/PMI/example_pPMI_0.csv', 
        'output_folder/03252021/PMI/example_pPMI_1.csv', 
        'output_folder/03252021/PMI/example_pPMI_2.csv', 
        'output_folder/03252021/PMI/example_ngrams.pkl', 
        'output_folder/03252021/PMI/example_nm1grams.pkl']
    '''
    sample = sentences
    
    with open('PMI/ngrams.pkl', 'r') as f:
        ngrams = pickle.load(f)
    
    
    with open('PMI/nm1grams.pkl', 'r') as f:
        nm1grams = pickle.load(f)
    
    N = 356033418959 # US american english v2 google ngrams
    nm1grams['_START_'] = float(sum([ ngrams[w] for w in ngrams.keys() if w[0] == '_START_']))
    
    
    def calc_prob(sentences, ngra=ngrams, nm1gra=nm1grams, ALPHA=0.1, lag=0):
        assert lag <= 2, 'impossible lag'
        results = []
        Z = len(ngrams.keys())*ALPHA + N
        for sent in sentences:
            string = sent[0]
            tokens = string.lower().split()
            mi = 0
            # No lag
            for t in range(0, len(tokens) - 1):
                joint_c = log(ngra[(tokens[t], tokens[t + 1])] + ngra[(tokens[t + 1], tokens[t])] + ALPHA)
                x_c = log(nm1gra[tokens[t]] + ALPHA * len(ngrams.keys()))
                y_c = log(nm1gra[tokens[t + 1]] + ALPHA * len(ngrams.keys()))
                pmi = max([0, (joint_c + log(Z) - x_c - y_c) / log(2)])
                mi += pmi
            # 1 word lag
            if lag >= 1:
                for t in range(0, len(tokens) - 2):
                    joint_c = log(ngra[(tokens[t], tokens[t + 2])] + ngra[(tokens[t + 2], tokens[t])] + ALPHA)
                    x_c = log(nm1gra[tokens[t]] + ALPHA * len(ngrams.keys()))
                    y_c = log(nm1gra[tokens[t + 2]] + ALPHA * len(ngrams.keys()))
                    pmi = max([0, (joint_c + log(Z) - x_c - y_c) / log(2)])
                    mi += pmi
            # 2 word lag
            if lag >= 2:
                for t in range(0, len(tokens) - 3):
                    joint_c = log(ngra[(tokens[t], tokens[t + 3])] + ngra[(tokens[t + 3], tokens[t])] + ALPHA)
                    x_c = log(nm1gra[tokens[t]] + ALPHA * len(ngrams.keys()))
                    y_c = log(nm1gra[tokens[t + 3]] + ALPHA * len(ngrams.keys()))
                    pmi = max([0,(joint_c + log(Z) - x_c - y_c) / log(2)])
                    mi += pmi
            results.append(','.join(sent[0].strip('\n'), str(mi)))
        return results
    
    
    result = calc_prob(sentences, lag=0)
    printstring = "\n".join(result)
    #with open('PMI/pPMI_0.csv', 'w') as f:
    with open(save_paths[0], 'w') as f:
        f.write(printstring)
    
    result = calc_prob(sentences, lag=1)
    printstring = "\n".join(result)
    #with open('PMI/pPMI_1.csv', 'w') as f:
    with open(save_paths[1], 'w') as f:
        f.write(printstring)
    
    result = calc_prob(sentences, lag=2)
    printstring = "\n".join(result)
    # with open('PMI/pPMI_2.csv', 'w') as f:
    with open(save_paths[2], 'w') as f:
        f.write(printstring)
########## End PMI Block

def sizeof_fmt(num, suffix='B'):
    '''
    This function can be used to print out how big a file is
    '''
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

