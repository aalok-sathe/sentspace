

import concurrent.futures
import hashlib
import math
import pdb
import pickle
from functools import partial
from itertools import chain
from time import time

# import seaborn as sns
import nltk
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.spatial.distance as ssd
from nltk import pos_tag
from scipy.stats import percentileofscore, zscore
from tqdm import tqdm

# from zs import ZS

_START_TIME = time()
def START_TIME(): return _START_TIME

# lemmas=WordNetLemmatizer()


# def import_franklin_sentences_set3(filename):
#     f2g = []
#     with open(filename, 'r') as file:
#         for line in file:
#             if line.startswith('passage') or line == '\n':
#                 pass
#             else:
#                 f2g.append(line.strip('\n').split())
                
#     return f2g

# def import_data(filename, dtype=None):
#     """
#     Import text file with \n after each line.
#     Pre-computed GloVe vectors can be loaded as:
#         glove_embed = import_data('../glove/vectors_243sentences.txt', dtype=lambda x: float(x))

#     """
#     f1g = []
#     with gzip.open(filename+'.zip', 'r') as file:
#         for line in file:
#             tokens = line.split()
#             if dtype:
#                 tokens = [dtype(token) for token in tokens]
#             f1g.append(tokens)
#     return f1g

# def get_wordlst(f1g):
#     """
#     Given list of sentences (each a list of tokens), return single list of tokens
#     """
#     wordlst = []
#     for sentence in f1g:
#         for word in sentence:
#             wordlst.append(word)
#     return wordlst

# def get_sent_num_passsage(f1g, lplst):
#     """
#     Given list of passage no. for each sentence, return sentence no. within passage (for each word)
#     """
#     sent_num = 0
#     snplst = []
#     current_label = lplst[0]
#     for sentence, label in zip(f1g, lplst):
#         sent_num += 1
#         if label != current_label:
#             sent_num = 1
#             current_label = label
#         for word in sentence:
#             snplst.append(sent_num)
#     return snplst



# download NLTK data if not already downloaded
def download_nltk_resources():
    for category, nltk_resource in [('taggers', 'averaged_perceptron_tagger'), 
                                    ('corpora', 'wordnet'),
                                    # ('tokenizers', 'punkt')
                                    ]:
        try:
            nltk.data.find(category+'/'+nltk_resource)
        except LookupError as e:
            try:
                nltk.download(nltk_resource)
            except FileExistsError:
                pass

def md5(fname) -> str:
    '''generates md5sum of the contents of fname
        fname (str): path to file whose md5sum we want
    '''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def sha1(ob):
    ob_repr = repr(ob)
    hash_object = hashlib.sha1()
    hash_object.update(ob_repr.encode('utf-8'))
    return hash_object.hexdigest()


def parallelize(function, *iterables, wrap_tqdm=True, desc='', **kwargs):
    """parallelizes a function by calling it on the supplied iterables and (static) kwargs.
       optionally wraps in tqdm for progress visualization 

    Args:
        function ([type]): [description]
        wrap_tqdm (bool, optional): [description]. Defaults to True.
        desc ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """    
    partialfn = partial(function, **kwargs)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        if wrap_tqdm:
            return [*tqdm(executor.map(partialfn, *iterables), total=len(iterables[0]), desc='[parallelized] '+desc)]
        return executor.map(partialfn, *iterables)

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



def merge_lists(list_a, list_b, feature=""):
    '''Input: Two lists with potentially missing values.
       Return: If list 1 contains NA vals, the NA val is replaced by the value in list 2 (either numerical val or np.nan again)
    '''
    # count_a, count_b, count_na = 0, 0, 0
    merged = []

    for val1, val2 in zip(list_a, list_b):
        merged += [val2 if np.isnan(val1) else val1]

        # if not np.isnan(val1):
        #     merged.append(val1)
        #     count_b += 1
        # else:
        #     merged.append(val2)
        #     if not np.isnan(val2):
        #         count_a += 1
        #     else:
        #         count_na += 1
    
    return merged

    n = len(merged)
    print(feature, f"| number of values derived from original form: {count_b}, {count_b/n*100:.2f}%")
    print(feature, f"| number of values derived from lemmatized form: {count_a}, {count_a/n*100:.2f}%")
    print(feature, f"| number of values = NA: {count_na}, {count_na/n*100:.2f}%")
    print('-'*79)


# def compile_results(wordlst, wordlst_l, wordlst_lem, taglst, is_content_lst, setlst,
#                     snlst, lplst_word, snplst, wnslst, catlst, wordlen, merged_vals):
# def compile_results(wordlst, wordlst_l, wordlst_lem, 
#                     taglst, is_content_lst, setlst,
#                     snlst, wordlen, merged_vals):
#     """
#     Return dataframe: each row is a word & its various associated values
#     """
#     result = pd.DataFrame({'Word': wordlst})
#     result['Word cleaned'] = wordlst_l
#     result['Word lemma'] = wordlst_lem

#     result['POS'] = taglst
#     result['Content/function'] = is_content_lst
#     result['Set no.'] = setlst
#     result['Sentence no.'] = snlst
#     #result['Passage no.'] = lplst_word
#     #result['Sentence no. within passage'] = snplst
#     #result['Word no. within sentence'] = wnslst
#     #result['Broad topic'] = catlst
#     result['Specific topic'] = ['']*len(wordlst)
#     result['Word length'] = wordlen
#     result['polysemy'] = merged_vals['polysemy']

#     # List what you want the columns to be called
#     cols = {'NRC_Arousal': 'Arousal', 
#             'NRC_Valence': 'Valence', 
#             'OSC': 'Orthography-Semantics Consistency', 
#             'aoa': 'Age of acquisition', 
#             'concreteness': 'Concreteness',
#             'lexical_decision_RT': 'Lexical decision RT',
#             'log_contextual_diversity': 'Contextual diversity (log)',
#             'log_lexical_frequency': 'Lexical frequency (log)',
#             'n_orthographic_neighbors': 'Frequency of orthographic neighbors',
#             'num_morpheme': 'Number of morphemes',
#             'prevalence': 'Prevalence',
#             'surprisal-3': 'Lexical surprisal',
#             'total_degree_centrality': 'Degree centrality',
#             'polysemy':'Polysemy',
#             'num_morpheme_poly':'Number of morphemes poly',
#             #'Pronoun Ratio':'Pronoun Ratio'
#     }

#     for key, val in cols.items():
#         result[val] = merged_vals[key]
#     return result
    

    
# def conform_word_lex_df_columns(df):
#     # List what you want the columns to be called
#     cols = {'NRC_Arousal': 'Arousal', 
#             'NRC_Valence': 'Valence', 
#             'OSC': 'Orthography-Semantics Consistency', 
#             'aoa': 'Age of acquisition', 
#             'concreteness': 'Concreteness',
#             'lexical_decision_RT': 'Lexical decision RT',
#             'log_contextual_diversity': 'Contextual diversity (log)',
#             'log_lexical_frequency': 'Lexical frequency (log)',
#             'n_orthographic_neighbors': 'Frequency of orthographic neighbors',
#             'num_morpheme': 'Number of morphemes',
#             'prevalence': 'Prevalence',
#             'surprisal-3': 'Lexical surprisal',
#             'total_degree_centrality': 'Degree centrality',
#             'polysemy':'Polysemy',
#             'num_morpheme_poly':'Number of morphemes poly',
#     }
#     df.rename(columns=cols)

#     # Remove empty column that are vestiges of temporary analyses
#     df = df.drop(columns=['Specific topic'])
#     return df

# def transform_features(df, method='default', cols_log=None, cols_z=None):
#     df = df.copy()
#     if method == 'default':
#         cols_log = ['Degree centrality', 'Frequency of orthographic neighbors']
#     if cols_log:
#         for col in cols_log:
#             df[col] = np.log10(df[col].astype('float')+1)
#             df = df.rename({col: col+' (log)'})
#         df = df.rename(columns={col: col+' (log)' for col in cols_log})
#     if cols_z:
#         for col in cols_z:
#             df[col] = zscore(df[col].astype('float'), nan_policy='omit')
#         df = df.rename(columns={col: col+' (z)' for col in cols_z})
#     return df
    
# #     return df_main

# def countNA(lst):
#     """
#     Return number of NAs in a list
#     """
#     return sum(np.isnan(lst))

# def countNA_df(df, features='all'):
#     """
#     Given dataframe of words and feature values
#     Return list of number of NAs in each word's features
#     """
#     if features == 'all':
#         features = ['Age of acquisition', 'Concreteness', 'Prevalence', 'Arousal', 'Valence', 'Dominance', 'Ambiguity: percentage of dominant', 'Log lexical frequency', 'Lexical surprisal', 'Word length']
#     df = df[features]
#     return list(df.isnull().sum(axis=1))

# def uniqueNA(df, feature):
#     """
#     Given dataframe of words and feature values & desired feature,
#     return set of unique words with NA in given feature
#     """
#     return sorted(set(df['Word cleaned'][df[feature].isna()]))

# def avgNA(result, feature):
#     """
#     Return fractions of words with NA (for given feature) in each sentence
#     """
#     return result.groupby('Sentence no.').apply(lambda data: countNA(data[feature])/len(data))

# def get_NA_words(result, wordlst_l, features):
#     """
#     Return list of words that have NA in at least one of the specified features
#     """
#     big_u_lst = []
#     for feature in features:
#         big_u_lst.extend(uniqueNA(result, feature))
#     u_lst = sorted(set(big_u_lst))
#     return (big_u_lst, u_lst)




# def avg_feature(data, feature, method):
#     """
#     Return average value of feature
#     """
#     if method=='strict':
#         data = data.dropna()
#     elif method=='all':
#         pass
#     else:
#         raise ValueError('Method not recognized')
#     return np.nanmean(np.array(data[feature], dtype=float))

# def get_sent_vectors(df, features, method='strict', content_only=False,
#                      save=False, save_path=None, **kwargs):
#     """
#     Return dataframe of sentence embeddings (each row as a sentence)
#     Method:
#         'strict' - if a word has NA in any feature, it is skipped in the sentence average for all features
#         'all' - use all non-NA values for sentence average in any feature
#     content_only - if True, use content words only in a sentence
#     """
#     pronoun_ratios = kwargs.get('pronoun_ratios', None)
#     content_ratios = kwargs.get('content_ratios', None)
    
#     if content_only:
#         df = df[df["Content/function"] == 1]
#     sent_vectors = pd.DataFrame({'Sentence no.': df['Sentence no.'].unique()})
#     df = df[features + ['Sentence no.']].groupby('Sentence no.')
#     for name, feature in zip(features, features):
#         sent_vectors[name] = list(df.apply(lambda data: avg_feature(data, feature, method)))
#     if pronoun_ratios is not None:
#         sent_vectors['Pronoun ratios'] = pronoun_ratios['pronoun_ratio']
#     # if content_ratios is not None:
#     #     sent_vectors['Content ratios'] = content_ratios['content_ratio']
#     if save:
#         sio.savemat(save_path, {'sent_vectors': sent_vectors.drop(columns=['Sentence no.']).to_numpy()})
#     return sent_vectors

# def get_differential_sents(embed1, embed2, n, result, method='euclidean'):
#     """
#     Print sentences with the largest distance between the two input embeddings
#     Return index of these sentences (return 1-indexed; assume sentence no. are 1-indexed)
#     """
#     if method == 'euclidean':
#         func = ssd.euclidean
#     elif method == 'correlation':
#         func = ssd.correlation
#     elif method == 'cosine':
#         func = ssd.cosine
#     else:
#         raise ValueError('Method not implemented')
#     diff = np.array([func(embed1[i], embed2[i]) for i in range(len(embed1))])
#     top_diff_ind = (-diff).argsort(axis=None)[:n]
#     top_diff_sent_no = [i+1 for i in top_diff_ind]
#     print('Sentences with largest differences:', top_diff_sent_no)
#     for i, idx in enumerate(top_diff_ind):
#         sent_no = idx+1
#         sent = result[result['Sentence no.'] == sent_no].sort_values('Word no. within sentence')
#         print(f'{i+1}, sentence {sent_no}: ', list(sent['Word']))
#         # print('Number of NA features for a word:', countNA_df(sent, features='all'))
#         # print(f'Value in embedding 1: {x[idx]}, embedding 2: {y[idx]}')
#         print(f'Distance: {diff[idx]}')
#         print()
#     return top_diff_sent_no






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

