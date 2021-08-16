import argparse
import json
from distutils.util import strtobool
from pathlib import Path
from numpy.core.numeric import outer

import pandas as pd
import sentspace.utils.io
from sentspace.syntax import get_features


def main(args):
    sentspace.utils.io.log(f'--- SYNTAX MODULE ---')

    # split = lambda s: s.split()

    # if Path(args.input).exists():
    #     with Path(args.input).open('r') as f:
    #         input = [*map(split, f.readlines())]
    # else:
    #     input = [*map(split, args.input.split('\n'))]

    # for i, item in enumerate(input):
    sentspace.utils.io.log(f'*** processing input ***')
    sentspace.utils.io.log(f'--- {args.input}')
    features = get_features(args.input, dlt=args.dlt, left_corner=args.left_corner,
                            parse_beam_width=args.parse_beam_width)        
    sentspace.utils.io.log(f'--- obtained features:')
    if args.output == 'pandas':
        print(pd.DataFrame(features, index=[0]))
    elif args.output == 'json':
        print(json.dumps(features, indent=4))
    elif args.output == 'dict':
        print(features)
    else:
        raise ValueError(f'output mode {args.output} not in supported modes (pandas, json)')


if __name__ == '__main__':

    # Parse input
    parser = argparse.ArgumentParser('sentspace.syntax')

    parser.add_argument('input', type=str, help='newline-separated sentences or file containing such sentences')
    parser.add_argument('-o', '--output', type=str, default='dict',
                        help='Output mode pandas/json/dict/yaml(not implemented)')

    # Add an option for a user to include their own stop words
    # parser.add_argument('-sw', '--stop_words', default=None,
    #                     help='Path to delimited file of words to filter out from analysis. For example example/stopwords.txt')

    # Add an option for a user to choose their benchmark
    # parser.add_argument('-b', '--benchmark', default='sentspace/benchmarks/lex/UD_corpora_lex_features_sents_all.csv',
    #                     help='Path to csv file of benchmark corpora For example benchmarks/lex/UD_corpora_lex_features_sents_all.csv')

    # Add an option for a user to choose to not do some analyses. Default is true
    parser.add_argument('-c', '--parse_beam_width', default=5_000, 
                        help='beam width to use with the Berkeley PCFG (smaller leads to faster output)')
    parser.add_argument('-dlt', '--dlt', type=strtobool,
                        default=True, help='calculate dependency locality theory (DLT) metric? [True]')
    parser.add_argument('-lc', '--left_corner', type=strtobool,
                        default=True, help='calculate left corner metric? [True]')

    args = parser.parse_args()
    sentspace.utils.io.log(f'SENTSPACE. Received arguments: {args}')
    main(args)
