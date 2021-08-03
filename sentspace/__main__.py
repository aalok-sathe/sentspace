# This file accepts as input a path to a file containing sentences and 
# prints out a csv with sentences per row and sentence features per column.

# print('Loading modules... (chunk 1/4)', end='\r')

import argparse
import os
import pathlib
import sys
from datetime import date
from distutils.util import strtobool

# print('Loading modules... (chunk 2/4)', end='\r')
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

# print('Loading modules... (chunk 3/4)', end='\r')
import sentspace.syntax
import sentspace.utils as utils
from sentspace.utils import wordnet
# from sentspace.utils.caching import cache_to_disk, cache_to_mem
# from sentspace.utils.text import get_flat_pos_tags
# from sentspace.utils.utils import wordnet

# print('Loading modules... (chunk 4/4)', end='\r')
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


def run_sentence_features_pipeline(input_file:pathlib.Path, stop_words_file:str=None, 
                                   benchmark_file: str = None, out_dir: str = None,
                                   lexical=False, syntax=False, embedding=False, semantic=False):
	"""runs the full sentence features pipeline on the given input according to
		requested submodules (lexical, syntax, ...).
		returns output in the form of TODO

	Args:
		input_file (str): path to input text file containing sentences
							one per line [required]
		stop_words_file (str): path to text file containing stopwords to filter
								out, one per line [optional]
		benchmark_file (str): path to a file containing a benchmark corpus to
								compare the current input against; e.g. UD [optional]
		
		{lexical,syntax,embedding,semantic,...} (bool): compute submodule features? [False]
	"""

	databases = utils.io.load_databases(features='all')


	# Sentence vectors settings
	embed_method = 'all' # options: 'strict', 'all'
	content_only = False

	# Define output paths
	default = True
	suffix = ''
	out_file_name = os.path.basename(input_file).split('.')[0]
	if default:
	    date_ = date.today().strftime('%m%d%Y')
	    output_folder = f'output_folder/{date_}'
	    sent_suffix = f"_{embed_method}"
	    if content_only:
	        sent_suffix = '_content' + sent_suffix
	    if stop_words_file is not None:
	        sent_suffix = '_content'+'_minus_stop_words' + sent_suffix
	    # Make out outlex path    
	    word_lex_output_path, embed_lex_output_path, plot_path, na_words_path, bench_perc_out_path = utils.io.create_output_path(output_folder, out_file_name, 'lex', sent_suffix=sent_suffix)
	    # Make output syntax path
	    _, sent_output_path = utils.io.create_output_path(output_folder, out_file_name, 'syntax', sent_suffix=sent_suffix)
	    glove_words_output_path, glove_sents_output_path = utils.io.create_output_path(output_folder, out_file_name, 'glove', sent_suffix=sent_suffix)
	    pmi_paths = utils.io.create_output_path(output_folder, out_file_name, 'PMI', sent_suffix=sent_suffix)
	    #pdb.set_trace()
	    lex_base = f'analysis_example/{date_}\\lex\\'
	else: # custom path
	    output_path = ''
	    plot_path = ''
	    na_words_path = ''
	    embed_output_path = ''

	token_lists, sentences = utils.io.read_sentences(input_file, stop_words_file=stop_words_file)
	flat_token_list = utils.text.get_flat_tokens(token_lists)
	flat_sentence_num = utils.text.get_flat_sentence_num(token_lists)
	flat_pos_tags = utils.text.get_flat_pos_tags(token_lists)
	flat_token_lens = utils.text.get_token_lens(flat_token_list)  # word length

	pronoun_ratios = utils.text.get_pronoun_ratio(flat_sentence_num, flat_pos_tags)
	word_num_list = utils.text.get_flat_word_num(token_lists)  # word no. within sentence

	nonletters = utils.text.get_nonletters(flat_token_list, exceptions=[]) # find all non-letter characters in file
	flat_cleaned_token_list = utils.text.strip_words(flat_token_list, method='punctuation', nonletters=nonletters) # clean words: strip nonletters/punctuation and lowercase
	flat_lemmatized_token_list = utils.text.get_lemmatized_tokens(flat_cleaned_token_list, flat_pos_tags, utils.text.pos_for_lemmatization)  # lemmas
	flat_is_content_word = utils.text.get_is_content(flat_pos_tags, content_pos=utils.text.pos_for_content) # content or function word


	surprisal_database = 'pickle/surprisal-3_dict.pkl' # default: 3-gram surprisal
	features_ignore_case = True
	features_transform = ['default', None, None] # format: [method, cols to log transform, cols to z-score] (if method is specified, the other two are ignored)

	# Get morpheme from polyglot library instead of library
	# TODO: where was this supposed to be used?
	# poly_morphemes = utils.get_poly_morpheme(flat_sentence_num, flat_token_list)

	
	# Word features
	n_words = len(flat_token_list)
	n_sentences = len(token_lists)
	# TODO what is this?
	setlst = [3] * n_words # set no. 
	print("Number of sentences:", n_sentences)
	print("Number of words:", n_words)
	print("Number of unique words (cleaned):", len(set(flat_cleaned_token_list)))
	print(f"Average number of words per sentence: {n_words/n_sentences:.2f}")
	print('-'*79)

	
	# Updated features
	#databases = utils.load_databases(ignore_case=features_ignore_case)
	if lexical == True:
		sentspace.lexical.get_lexical_features(None)
		exit()
	
	if syntax:
		exit()
		# Get content ratio
		content_ratios = sentspace.syntax.get_content_ratio(flat_sentence_num, flat_is_content_word)

		syn_feats = sentspace.syntax.get_tree_features('hello my name is syntax', dlt=True, left_corner=True)
		print(syn_feats)

		syntax_df = pd.DataFrame(
			{'Sentence no.': content_ratios['sent_num'], 'Content Ratio': content_ratios['content_ratio']})
		syntax_df.to_csv(sent_output_path, index=False)


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
	
	if embedding == True:
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
	features = run_sentence_features_pipeline(args.input_file, stop_words_file=args.stop_words, 
											  benchmark_file=args.benchmark, lexical=args.lexical, 
											  syntax=args.syntax, embedding=args.embedding,
											  semantic=args.semantic,
											  out_dir=args.out_dir)
	#estimate_sentence_embeddings(args.input_file)


if __name__ == "__main__":

	# Parse input
	parser = argparse.ArgumentParser('sentspace', 
                             	     usage="""
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

	parser.add_argument('-i', '--input_file', type=str,
						default=None,
						required=True,
						help='path to input file or a single sentence. If '
						     'supplying a file, it must be .csv .txt or .xlsx,'
							 ' e.g., example/example.csv')

	# Add an option for a user to include their own stop words
	parser.add_argument('-sw','--stop_words', default=None, type=str, 
						help='path to delimited file of words to filter out from analysis, e.g., example/stopwords.txt')
	
	# Add an option for a user to choose their benchmark
	parser.add_argument('-b', '--benchmark', type=str,
						default='sentspace/benchmarks/lexical/UD_corpora_lex_features_sents_all.csv',
                     	help='path to csv file of benchmark corpora For example benchmarks/lexical/UD_corpora_lex_features_sents_all.csv')

	# parser.add_argument('--cache_dir', default='.cache', type=str,
    #                  	help='path to directory where results may be cached')

	parser.add_argument('-o', '--out_dir', default='./sentspace_output', type=str,
                     	help='path to output directory where results may be stored')

	# Add an option for a user to choose to not do some analyses. Default is true
	parser.add_argument('-lx','--lexical', type=strtobool, default=True, help='compute lexical features? [True]')
	parser.add_argument('-sx','--syntax', type=strtobool, default=True, help='compute syntactic features? [True]')
	parser.add_argument('-em','--embedding', type=strtobool, default=True, help='compute sentence embeddings? [True]')
	parser.add_argument('-sm','--semantic', type=strtobool, default=True, help='compute semantic (multi-word) features? [True]')
	
	args = parser.parse_args()	
	print('-'*79)
	print('Received arguments:', args)
	print('-'*79)

	main(args)
