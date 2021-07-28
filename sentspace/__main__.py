# This file accepts as input a path to a file containing sentences and 
# prints out a csv with sentences per row and sentence features per column.

print('Loading modules...')

import argparse
import os
import sys
from datetime import date
from distutils.util import strtobool

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import wordnet

import sentspace.utils as utils
from sentspace.sanity_checks import sanity_check_databases
import sentspace.syntax

import matplotlib.pyplot as plt; plt.rcdefaults()

# download NLTK data if not already downloaded
for nltk_resource in ['taggers/averaged_perceptron_tagger', 'corpora/wordnet']:
	try:
		nltk.data.find(nltk_resource)
	except LookupError as e:
		nltk.download(nltk_resource)

### supposedly unused imports ###

# import pickle
# import scipy.io as sio
# from itertools import chain
# import itertools
# import string
# from collections import Counter, defaultdict

# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk import word_tokenize
# from nltk import parse

# import copy
# import simmat_util as sim
# from importlib import reload 
# from sklearn.decomposition import PCA
# from sklearn.manifold import MDS
# from scipy.stats import zscore
# from s3 import load_feature


def estimate_sentence_embeddings(path_2_sentence_file, path_2_stop_words, path_2_bechmark,
								 calc_lex, calc_syntax, calc_glove,**kwargs):
								 	
	#pdb.set_trace()
	stop_words = None
	if path_2_stop_words:
		stop_words = np.loadtxt(path_2_stop_words, delimiter='\t', unpack=False, dtype=str)

	print('Loading feature databases...')
	databases = utils.load_databases(
		features=['NRC_Arousal', 'NRC_Valence', 'OSC', 'aoa', 'concreteness',
				  'lexical_decision_RT', 'log_contextual_diversity', 
				  'log_lexical_frequency', 'n_orthographic_neighbors', 'num_morpheme',
				  'prevalence', 'surprisal-3', 'total_degree_centrality'])

	print('Performing sanity checks...')
	sanity_check_databases()(databases)

	print('Parsing input sentences...')
	plot_dist = False
	if plot_dist == True:
		features = []
		values = []
		for feat, d in databases.items():
		    values.extend(d.values())
		    features.extend([feat]*len(d))
		df = pd.DataFrame({'feature': features, 'value': values})
		print('Preparing plots')
		g = sns.FacetGrid(df, col='feature', col_wrap=3, sharex=False, sharey=False)
		g.map(sns.distplot, 'value')
		plt.show()
		
	# Sentence vectors settings
	embed_method = 'all' # options: 'strict', 'all'
	content_only = False

	# Define output paths
	default = True
	suffix = ''
	out_file_name = os.path.basename(path_2_sentence_file).split('.')[0]
	if default:
	    date_ = date.today().strftime('%m%d%Y')
	    output_folder = f'output_folder/{date_}'
	    sent_suffix = f"_{embed_method}"
	    if content_only:
	        sent_suffix = '_content' + sent_suffix
	    if path_2_stop_words is not None:
	        sent_suffix = '_content'+'_minus_stop_words' + sent_suffix
	    # Make out outlex path    
	    word_lex_output_path, embed_lex_output_path, plot_path, na_words_path, bench_perc_out_path = utils.create_output_path(output_folder, out_file_name, 'lex', sent_suffix=sent_suffix)
	    # Make output syntax path
	    _, sent_output_path = utils.create_output_path(output_folder, out_file_name, 'syntax', sent_suffix=sent_suffix)
	    glove_words_output_path, glove_sents_output_path = utils.create_output_path(output_folder, out_file_name, 'glove', sent_suffix=sent_suffix)
	    pmi_paths = utils.create_output_path(output_folder, out_file_name, 'PMI', sent_suffix=sent_suffix)
	    #pdb.set_trace()
	    lex_base = f'analysis_example/{date_}\\lex\\'
	else: # custom path
	    output_path = ''
	    plot_path = ''
	    na_words_path = ''
	    embed_output_path = ''

	# Prepare variables
	sent_path = path_2_sentence_file # sentence file
	sent_tokens, sent_rows, _ = utils.import_sentences(sent_path, sent_only = True,stop_words=stop_words)
	setlst = None # list of set no.
	
	lemmatized_pos = [wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV] # define POS used for lemmatization
	content_pos = [wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV] # define POS that count as content words
	clean_word_method = 'punctuation' 
	surprisal_database = 'pickle/surprisal-3_dict.pkl' # default: 3-gram surprisal
	features_ignore_case = True
	features_transform = ['default', None, None] # format: [method, cols to log transform, cols to z-score] (if method is specified, the other two are ignored)

	# Word features
	wordlst = utils.get_wordlst(sent_tokens) # raw words
	taglst = utils.get_pos_tag(sent_tokens) # POS tag
	numWords = len(wordlst)
	numSentences = len(sent_tokens)
	setlst = [3]*numWords # set no. 
	sent_num_list = utils.get_sent_num(sent_tokens) # sentence no.

	# Get pronoun ratio
	pronoun_ratios = utils.get_pronoun_ratio(sent_num_list, taglst)

	# Get morpheme from polyglot library instead of library
	poly_morphemes = utils.get_poly_morpheme(sent_num_list, wordlst)

	word_num_list = utils.get_word_num(sent_tokens) # word no. within sentence

	nonletters = utils.get_nonletters(wordlst, exceptions=[]) # find all non-letter characters in file
	wordlst_l = utils.strip_words(wordlst, method=clean_word_method, nonletters=nonletters) # clean words: strip nonletters/punctuation and lowercase
	print("Number of sentences:", numSentences)
	print("Number of words:", numWords)
	print("Number of unique words (cleaned):", len(set(wordlst_l)))
	print(f"Average number of words per sentence: {numWords/numSentences:.2f}")
	print(utils.get_divider())

	wordlen = utils.get_wordlen(wordlst_l) # word length
	wordlst_lem = utils.get_lemma(wordlst_l, taglst, lemmatized_pos) # lemmas
	
	# Get is content boolean
	is_content_lst = utils.get_is_content(taglst, content_pos=content_pos) # content or function word
	
	# Updated features
	#databases = utils.load_databases(ignore_case=features_ignore_case)
	if calc_lex == True:
		print('Estimating lexical and syntactical features')
		
		merged_vals = utils.get_all_features_merged(wordlst_l, wordlst_lem, databases) # lexical features
		# Clear variables so we have RAM
		del databases
		
		# Results
		result = utils.compile_results(wordlst, wordlst_l, wordlst_lem, 
										taglst, is_content_lst, setlst, 
										sent_num_list, wordlen, merged_vals)
	
		result = utils.transform_features(result, *features_transform)
	
		features_used = ['Age of acquisition', 'Arousal', 'Concreteness', 
						'Contextual diversity (log)', 'Degree centrality (log)', 
						'Frequency of orthographic neighbors (log)', 'Lexical decision RT', 
						'Lexical frequency (log)', 'Lexical surprisal', 'Number of morphemes', 'Number of morphemes poly',
						'Orthography-Semantics Consistency', 'Prevalence', 'Valence', 'Word length','Polysemy']
	
		print('Computing sentence embeddings')
		sent_embed = utils.get_sent_vectors(result, features_used, embed_method, 
		                                   content_only=content_only, 
		                                   pronoun_ratios=pronoun_ratios,
		                                   )
		lex_per_word_with_uniform_column = utils.conform_word_lex_df_columns(result)
		lex_per_word_with_uniform_column.to_csv(word_lex_output_path,index=False)
		
		print('Writing lex sentence embedding to csv at '+ embed_lex_output_path)
		sent_embed.to_csv(embed_lex_output_path, index=False)
	
		# Make the syntax excel sheet
		print('Writing syntax sentence embedding to csv at '+ sent_output_path)
		#pdb.set_trace()
		
		# Read in benchmark data
		df_benchmark = pd.read_csv(path_2_bechmark)
	
		# Return percentile per sentence for each 
		percentile_df = utils.return_percentile_df(df_benchmark, sent_embed)
		print('Writing percentiles')
		percentile_df.to_csv(bench_perc_out_path,index=False)
	
	if calc_syntax:
		# Get content ratio
		content_ratios = utils.get_content_ratio(sent_num_list, is_content_lst)

		syn_feats = sentspace.syntax.get_features('hello my name is syntax', dlt=True, left_corner=True)
		print(syn_feats)

		syntax_df = pd.DataFrame(
			{'Sentence no.': content_ratios['sent_num'], 'Content Ratio': content_ratios['content_ratio']})
		syntax_df.to_csv(sent_output_path, index=False)


	# DEBUG
	exit()

	# Clear vars for RAM
	if 'databases' in locals():
		del databases
	if 'df' in locals():
		del df
	if 'sent_embed' in locals():
		del sent_embed
	if 'syntax_df' in locals():
		del syntax_df
	if 'lex_per_word_with_uniform_column' in locals():
		del lex_per_word_with_uniform_column
	if 'merged_vals' in locals():
		del merged_vals
	if 'content_ratios' in locals():	
		del content_ratios
	
	# Calculate PMI
	# utils.GrabNGrams(sent_rows,pmi_paths)
	# utils.pPMI(sent_rows, pmi_paths)
	# pdb.set_trace()
	
	if calc_glove == True:
		# Get GloVE
		result = utils.compile_results_for_glove_only(wordlst, wordlst_l, wordlst_lem, 
										taglst, is_content_lst, setlst, 
										sent_num_list, wordlen)
		del wordlst
		del wordlst_l
		del wordlst_lem
		del taglst
		del is_content_lst
		del setlst
		del sent_num_list
		del wordlen
		
		result['Word no. within sentence'] = word_num_list
		sent_version = utils.get_sent_version('cleaned', result)
		vocab = utils.get_vocab(sent_version)
		
		for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
			key= lambda x: -x[1])[:10]:
			print("{:>30}: {:>8}".format(name, utils.sizeof_fmt(size)))
			
		print('Extracting Glove representations from vocabulary')
		word2vects = utils.read_glove_embed(vocab, 'Glove/glove.840B.300d.txt')
		print('Extracting glove for each word')
		glove_words = utils.get_glove_word(sent_version, word2vects)
		print('Extracting glove for each sentence')
		glove_sents = utils.get_glove_sent(glove_words)
		
		# Save Glove
		print('Writing Glove embeddings at '+glove_words_output_path)
		glove_words.to_csv(glove_words_output_path, index=False)
		glove_sents.to_csv(glove_sents_output_path, index=False)


	# Plot input data to benchmark data
	#utils.plot_usr_input_against_benchmark_dist_plots(df_benchmark, sent_embed)

def main(args):
	'''used to run the main pipeline, start to end, depending on the arguments and flags
	'''

	# Estimate sentence embeddings
	estimate_sentence_embeddings(args.input_file, path_2_stop_words=args.stop_words, path_2_bechmark=args.benchmark,
                              calc_lex=args.calc_lex, calc_syntax=args.calc_syntax, calc_glove=args.calc_glove)
	#estimate_sentence_embeddings(args.input_file)


if __name__ == "__main__":

	# Parse input
	parser = argparse.ArgumentParser('sentspace', 
                             	     usage=
		"""
		Example run:
			python3 -m sentspace -i example/example.csv

		Example run without Glove estimation
			python3 -m sentspace -i example/example.csv --glove false
			
		Example run without lexical feature estimation
			python3 -m sentspace -i example/example.csv --lexical false
			
		Example	run with stop words:
			python3 -m sentspace -i example/example.csv --stop_words example/stopwords.txt
		"""
    )

	parser.add_argument('-i', '--input_file', nargs='?', type=argparse.FileType('r'), 
						default=sys.stdin,
						help='Path to input file or a single sentence. If '
						     'supplying a file, it must be .csv .txt or .xlsx,'
							 ' e.g., example/example.csv')

	# Add an option for a user to include their own stop words
	parser.add_argument('-sw','--stop_words', default=None, 
						help='Path to delimited file of words to filter out from analysis, e.g., example/stopwords.txt')
	
	# Add an option for a user to choose their benchmark
	parser.add_argument('-b', '--benchmark', default='sentspace/benchmarks/lex/UD_corpora_lex_features_sents_all.csv', 
						help='Path to csv file of benchmark corpora For example benchmarks/lex/UD_corpora_lex_features_sents_all.csv')

	# Add an option for a user to choose to not do some analyses. Default is true
	parser.add_argument('-l','--lexical', type = strtobool, default=True, help='calculate lexical features? [True]')
	parser.add_argument('-s','--syntax', type = strtobool, default=True, help='calculate syntactic features? [True]')
	parser.add_argument('-g','--glove', type = strtobool, default=True, help='compute glove embeddings? [True]')
	
	args = parser.parse_args()	
	print(args)

	main(args)
