# This file accepts as input a path to a file containing sentences and 
# prints out a csv with sentences per row and sentence features per column.

# print('Loading modules... (chunk 1/4)', end='\r')

from distutils.util import strtobool
import argparse
import pathlib
import sys
# import numpy as np

# print('Loading modules... (chunk 2/4)', end='\r')
import nltk
import pandas as pd
import seaborn as sns

# print('Loading modules... (chunk 3/4)', end='\r')
import sentspace.syntax
import sentspace.utils as utils
from tqdm import tqdm
from sentspace.utils import wordnet
import json

# from sentspace.utils.caching import cache_to_disk, cache_to_mem
# from sentspace.utils.text import get_flat_pos_tags
# from sentspace.utils.utils import wordnet



def run_sentence_features_pipeline(input_file: str, stop_words_file: str = None,
                                   benchmark_file: str = None, output_dir: str = None,
                                   lexical: bool = False, syntax: bool = False, 
								   embedding: bool = False, semantic: bool = False):
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

	# create output folder
	utils.io.log('creating output folder')
	# (sent_output_path, 
	#  glove_words_output_path, 
	#  glove_sents_output_path) 
	output_dir = utils.io.create_output_paths(input_file,
	 															 output_dir=output_dir,
                                                                 stop_words_file=stop_words_file)
	with (output_dir / 'config.txt').open('w+') as f:
		print(args, file=f)

	utils.io.log('reading input sentences')
	token_lists, sentences = utils.io.read_sentences(input_file, stop_words_file=stop_words_file)
	utils.io.log('---done--- reading input sentences')

	# flat_token_list = utils.text.get_flat_tokens(token_lists)
	# flat_sentence_num = utils.text.get_flat_sentence_num(token_lists)
	# flat_pos_tags = utils.text.get_flat_pos_tags(token_lists)
	# flat_token_lens = utils.text.get_token_lens(flat_token_list)  # word length

	# pronoun_ratios = utils.text.get_pronoun_ratio(flat_sentence_num, flat_pos_tags)
	# word_num_list = utils.text.get_flat_word_num(token_lists)  # word no. within sentence

	# nonletters = utils.text.get_nonletters(flat_token_list, exceptions=[]) # find all non-letter characters in file
	# flat_cleaned_token_list = utils.text.strip_words(flat_token_list, method='punctuation', nonletters=nonletters) # clean words: strip nonletters/punctuation and lowercase
	# flat_lemmatized_token_list = utils.text.get_lemmatized_tokens(flat_cleaned_token_list, flat_pos_tags, utils.text.pos_for_lemmatization)  # lemmas
	# flat_is_content_word = utils.text.get_is_content(flat_pos_tags, content_pos=utils.text.pos_for_content) # content or function word


	# surprisal_database = 'pickle/surprisal-3_dict.pkl' # default: 3-gram surprisal
	# features_ignore_case = True
	# features_transform = ['default', None, None] # format: [method, cols to log transform, cols to z-score] (if method is specified, the other two are ignored)

	# Get morpheme from polyglot library instead of library
	# TODO: where was this supposed to be used?
	# poly_morphemes = utils.get_poly_morpheme(flat_sentence_num, flat_token_list)

	
	# Word features
	# n_words = len(flat_token_list)
	# n_sentences = len(token_lists)
	# # TODO what is this?
	# setlst = [3] * n_words # set no. 
	# print("Number of sentences:", n_sentences)
	# print("Number of words:", n_words)
	# print("Number of unique words (cleaned):", len(set(flat_cleaned_token_list)))
	# print(f"Average number of words per sentence: {n_words/n_sentences:.2f}")
	# print('-'*79)

	
	# Updated features
	#databases = utils.load_databases(ignore_case=features_ignore_case)
	if lexical:
		utils.io.log('*** running lexical submodule pipeline')
		lexical_features = [sentspace.lexical.get_features(sentence) 
							for sentence in tqdm(token_lists, desc='Lexical pipeline')]
		
		lexical_out = output_dir / 'lexical'
		lexical_out.mkdir(parents=True, exist_ok=True)
		with (lexical_out/'features.json').open('w') as f:
			json.dump(lexical_features, f)
		
	
	if syntax:
		utils.io.log('*** running syntax submodule pipeline')
		syntax_features = sentspace.syntax.get_features(args.input_file, dlt=True, left_corner=True)

		syntax_out = output_dir / 'syntax'
		syntax_out.mkdir(parents=True, exist_ok=True)
		with (syntax_out/'features.json').open('w') as f:
			f.write(str(syntax_features))

		# syntax_df = pd.DataFrame(
		# 	{'Sentence no.': content_ratios['sent_num'], 'Content Ratio': content_ratios['content_ratio']})
		# syntax_df.to_csv(sent_output_path, index=False)


	# Calculate PMI
	# utils.GrabNGrams(sent_rows,pmi_paths)
	# utils.pPMI(sent_rows, pmi_paths)
	# pdb.set_trace()
	
	if embedding:
		utils.io.log('*** running embedding submodule pipeline')
		return
		# Get GloVE
		result = utils.compile_results_for_glove_only(wordlst, wordlst_l, wordlst_lem,
													  taglst, is_content_lst, setlst,
												      sent_num_list, wordlen)

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
											  output_dir=args.output_dir)
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

	parser.add_argument('-o', '--output_dir', default='./out', type=str,
                     	help='path to output directory where results may be stored')

	# Add an option for a user to choose to not do some analyses. Default is true
	parser.add_argument('-lx','--lexical', type=strtobool, default=True, help='compute lexical features? [True]')
	parser.add_argument('-sx','--syntax', type=strtobool, default=True, help='compute syntactic features? [True]')
	parser.add_argument('-em','--embedding', type=strtobool, default=True, help='compute sentence embeddings? [True]')
	parser.add_argument('-sm','--semantic', type=strtobool, default=True, help='compute semantic (multi-word) features? [True]')
	
	args = parser.parse_args()	
	utils.io.log(f'SENTSPACE. Received arguments: {args}')
	main(args)
