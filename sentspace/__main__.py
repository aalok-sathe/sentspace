#!/usr/bin/env python

import argparse
import json
import pathlib
import sys
from distutils.util import strtobool

import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sentspace
import sentspace.utils as utils
from itertools import chain
from functools import reduce, partial
# import numpy as np

import multiprocessing



#     # Plot input data to benchmark data
#     #utils.plot_usr_input_against_benchmark_dist_plots(df_benchmark, sent_embed)

def main(args):
    '''used to run the main pipeline, start to end, depending on the arguments and flags
    '''
    
    utils.io.log(f'SENTSPACE. Received arguments: {args}')

    # Estimate sentence embeddings
    output_dir = sentspace.run_sentence_features_pipeline(args.input_file, stop_words_file=args.stop_words,
                                                          benchmark_file=args.benchmark, process_lexical=args.lexical,
                                                          process_syntax=args.syntax, process_embedding=args.embedding,
                                                          process_semantic=args.semantic,
                                                          output_dir=args.output_dir,
                                                          output_format=args.output_format,
                                                          parallelize=args.parallelize,
                                                          # TODO: return_df or return_path?
                                                          emb_data_dir=args.emb_data_dir)

    with (output_dir/'FINISHED').open('w+') as f:
        pass

if __name__ == "__main__":

    # Parse input
    parser = argparse.ArgumentParser('sentspace', 
                                      usage="""
                                            """
    )

    parser.add_argument('input_file', type=str,
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

    parser.add_argument('-p', '--parallelize', default=True, type=strtobool, 
                        help='use multiple threads to compute features? '
                             'disable using `-p False` in case issues arise.')

    parser.add_argument('-o', '--output_dir', default='./out', type=str,
                         help='path to output directory where results may be stored')

    parser.add_argument('-of', '--output_format', default='pkl', type=str,
                        choices=['pkl','tsv'])

    # Add an option for a user to choose to not do some analyses. Default is true
    parser.add_argument('-lex','--lexical', type=strtobool, default=False, help='compute lexical features? [False]')
    parser.add_argument('-syn','--syntax', type=strtobool, default=False, help='compute syntactic features? [False]')
    parser.add_argument('-emb','--embedding', type=strtobool, default=False, help='compute high-dimensional sentence representations? [False]')
    parser.add_argument('-sem','--semantic', type=strtobool, default=False, help='compute semantic (multi-word) features? [False]')
    
    parser.add_argument('--emb_data_dir', default='/om/data/public/glove/', type=str,
                         help='path to output directory where results may be stored')
    # parser.add_argument('--cache_dir', default=)

    args = parser.parse_args()	
    main(args)
