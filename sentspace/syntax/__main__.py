import argparse
import json
from distutils.util import strtobool

import pandas as pd
from sentspace.syntax import get_features
import sentspace.utils.io


def main(args):

    split = lambda s: s.split()

    if Path(args.input).exists():
        with Path(args.input).open('r') as f:
            input = [*map(split, f.readlines())]
    else:
        input = [*map(split, args.input.split('\n'))]

    for i, item in enumerate(input):
        sentspace.utils.io.log(f'*** processing sentence {i}/{len(input)} ***')
        sentspace.utils.io.log(f'--- {item}')
        features = get_features(item, dlt=args.dlt, left_corner=args.left_corner)        
        sentspace.utils.io.log(f'--- obtained features:')
        if args.output == 'pandas':
            print(pd.DataFrame(features))
        elif args.output == 'json':
            print(json.dumps(features, indent=4))
        else:
            raise ValueError(f'output mode {args.output} not in supported modes (pandas, json)')


if __name__ == '__main__':

    # Parse input
    parser = argparse.ArgumentParser('sentspace.syntax')

    parser.add_argument('input', type=str, help='File or newline-separated sentences to process')
    parser.add_argument('-o', '--output', type=str, default='pandas',
                        help='Output mode pandas/json/yaml(not implemented)')

    # Add an option for a user to include their own stop words
    # parser.add_argument('-sw', '--stop_words', default=None,
    #                     help='Path to delimited file of words to filter out from analysis. For example example/stopwords.txt')

    # Add an option for a user to choose their benchmark
    # parser.add_argument('-b', '--benchmark', default='sentspace/benchmarks/lex/UD_corpora_lex_features_sents_all.csv',
    #                     help='Path to csv file of benchmark corpora For example benchmarks/lex/UD_corpora_lex_features_sents_all.csv')

    # Add an option for a user to choose to not do some analyses. Default is true
    parser.add_argument('-dlt', '--dlt', type=strtobool,
                        default=True, help='calculate dependency locality theory (DLT) metric? [True]')
    parser.add_argument('-lc', '--left_corner', type=strtobool,
                        default=True, help='calculate left corner metric? [True]')

    args = parser.parse_args()
    print(args)
    
    main(args)
