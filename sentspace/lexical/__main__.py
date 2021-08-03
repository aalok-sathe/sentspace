import argparse
from sentspace.lexical import get_features


def main(args):
    
    print('Estimating lexical features')



if __name__ == '__main__':
    # Parse input
    parser = argparse.ArgumentParser('sentspace.lexical')

    parser.add_argument('input', type=str, #argparse.FileType('r'), 
                        # default=sys.stdin,
                        help='File (currently NotImplemented) or single sentence to process')

    # Add an option for a user to include their own stop words
    parser.add_argument('-sw', '--stop_words', default=None,
                        help='Path to delimited file of words to filter out from analysis. For example example/stopwords.txt')

    # Add an option for a user to choose their benchmark
    parser.add_argument('-b', '--benchmark', default='sentspace/benchmarks/lex/UD_corpora_lex_features_sents_all.csv',
                        help='Path to csv file of benchmark corpora For example benchmarks/lex/UD_corpora_lex_features_sents_all.csv')

    # Add an option for a user to choose to not do some analyses. Default is true
    parser.add_argument('-dlt', '--dlt', type=strtobool,
                        default=True, help='calculate dependency locality theory (DLT) metric? [True]')
    parser.add_argument('-lc', '--left_corner', type=strtobool,
                        default=True, help='calculate left corner metric? [True]')

    args = parser.parse_args()
    print(args)
    
    main(args)
