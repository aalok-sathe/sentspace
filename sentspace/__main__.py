#!/usr/bin/env python3

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

    # Estimate sentence embeddings
    features = sentspace.run_sentence_features_pipeline(args.input_file, stop_words_file=args.stop_words,
                                                        benchmark_file=args.benchmark, lexical=args.lexical,
                                                        syntax=args.syntax, embedding=args.embedding,
                                                        semantic=args.semantic,
                                                        output_dir=args.output_dir,
                                                        output_format=args.output_format,
                                                        #
                                                        emb_data_dir=args.emb_data_dir)
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

    parser.add_argument('-o', '--output_dir', default='./out', type=str,
                         help='path to output directory where results may be stored')

    parser.add_argument('-of', '--output_format', default='pkl', type=str,
                        choices=['pkl','tsv'])

    # Add an option for a user to choose to not do some analyses. Default is true
    parser.add_argument('-lex','--lexical', type=strtobool, default=True, help='compute lexical features? [True]')
    parser.add_argument('-syn','--syntax', type=strtobool, default=True, help='compute syntactic features? [True]')
    parser.add_argument('-emb','--embedding', type=strtobool, default=True, help='compute sentence embeddings? [True]')
    parser.add_argument('-sem','--semantic', type=strtobool, default=True, help='compute semantic (multi-word) features? [True]')
    
    parser.add_argument('--emb_data_dir', default='/om/data/public/glove/', type=str,
                         help='path to output directory where results may be stored')

    args = parser.parse_args()	
    utils.io.log(f'SENTSPACE. Received arguments: {args}')
    main(args)
