import pandas as pd

from fawoc.terms import Label
from slrkit_utils.argument_parser import ArgParse
from utils import assert_column


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: ArgParse
    """
    parser = ArgParse(description='Extracts the stopwords classified '
                                  'with FAWOC')
    parser.add_argument('terms_file', type=str,
                        help='path to the terms file',
                        input=True, cli_only=True)
    parser.add_argument('outfile', type=str,
                        help='path to the stopwords output file',
                        output=True, cli_only=True)
    return parser


def main():
    args = init_argparser().parse_args()
    stopword_extractor(args)


def stopword_extractor(args):
    df = pd.read_csv(args.terms_file, sep='\t')
    assert_column(args.terms_file, df, ['term', 'label'])
    stopwords = df[df['label'] == Label.STOPWORD.label_name]
    with open(args.outfile, 'w') as file:
        for _, row in stopwords.iterrows():
            print(row['term'], file=file)


if __name__ == "__main__":
    main()
