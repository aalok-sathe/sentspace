from pathlib import Path
import argparse
import pickle

import pandas as pd


def main(**kwargs):
    """used to run the main pipeline, start to end, depending on the arguments and flags"""
    # Parse input
    parser = argparse.ArgumentParser("sentspace")

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file to package as a lexical feature e.g., example/example.csv",
    )
    parser.add_argument(
        "-word",
        "--word_column",
        default="Word",
        help="Column we should extract the word from (default: 'Word')",
    )
    parser.add_argument(
        "-norm",
        "--norm_column",
        required=True,
        help="Column we should extract the norm from",
    )
    parser.add_argument(
        "-name",
        "--norm_name",
        required=False,
        help="Name of the feature to package. By default, the same as `norm_column`",
    )
    parser.add_argument(
        "-out",
        "--output_directory",
        default="~/.cache/sentspace/",
        help="Where should we output the `norm_name.pkl` packaged file?",
    )

    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.input_file)
    words = df[args.word_column]
    norms = df[args.norm_column]
    mapping = dict(zip(words, norms))
    name = args.norm_name or args.norm_column

    out = Path(args.output_directory).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.pkl").open("wb") as f:
        pickle.dump(mapping, f)


if __name__ == "__main__":
    main()
