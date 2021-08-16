import argparse
import json
from sentspace.lexical import get_features
import sentspace.utils.io
from distutils.util import strtobool
import pandas as pd
from pathlib import Path

def main(args):
    sentspace.utils.io.log(f'--- LEXICAL MODULE ---')

    split = lambda s: s.split()

    if Path(args.input).exists():
        with Path(args.input).open('r') as f:
            input = [*map(split, f.readlines())]
    else:
        input = [*map(split, args.input.split('\n'))]

    for i, item in enumerate(input):
        sentspace.utils.io.log(f'*** processing sentence {i}/{len(input)} ***')
        sentspace.utils.io.log(f'--- {item}')
        features = get_features(item)
        sentspace.utils.io.log(f'--- obtained features:')
        if args.output == 'pandas':
            print(pd.DataFrame(features))
        elif args.output == 'json':
            print(json.dumps(features, indent=4))
        else:
            raise ValueError(f'output mode {args.output} not in supported modes (pandas, json)')


if __name__ == '__main__':
    # Parse input
    parser = argparse.ArgumentParser('sentspace.lexical')

    parser.add_argument('input', type=str, help='File or newline-separated sentences to process')
    parser.add_argument('-o', '--output', type=str, default='pandas',
                        help='Output mode pandas/json/yaml(not implemented)')

    # Add an option for a user to include their own stop words
    # parser.add_argument('-sw', '--stop_words', default=None,
    #                     help='Path to delimited file of words to filter out from analysis. For example example/stopwords.txt')

    # Add an option for a user to choose their benchmark
    # parser.add_argument('-b', '--benchmark', default='sentspace/benchmarks/lex/UD_corpora_lex_features_sents_all.csv',
    #                     help='Path to csv file of benchmark corpora For example benchmarks/lex/UD_corpora_lex_features_sents_all.csv')

    args = parser.parse_args()
    sentspace.utils.io.log(f'SENTSPACE. Received arguments: {args}')
    main(args)
