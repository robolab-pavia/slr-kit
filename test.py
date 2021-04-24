import argparse

from utils import AppendMultipleFilesAction


class AppendMultiplePairsAction(argparse.Action):
    """
    Action for argparse that collects multiple option arguments as a set

    This can be used to implement a option that can have multiple arguments.
    The option itself may be given multiple time on the command line.
    All the arguments are collected in a set of string.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            if ((isinstance(nargs, str) and nargs in ['*', '?'])
                    or (isinstance(nargs, int) and nargs < 2)):
                raise ValueError(f'nargs = {nargs} is not allowed')

        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        pairs = getattr(namespace, self.dest, None)
        if pairs is None:
            pairs = []

        if not isinstance(values, list):
            pair = (values, None)
            length = 1
        else:
            length = len(values)
            if length == 1:
                pair = (values[0], None)
            else:
                pair = tuple(values)

        if length > 2:
            parser.error(f'Arguments of {self.option_strings} must be at most 2')
        
        if self.nargs == 2 and length != 2:
            parser.error(f'Too few arguments for {self.option_strings}')

        pairs.append(pair)

        setattr(namespace, self.dest, pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', '-p', action=AppendMultiplePairsAction, nargs='+')
    args = parser.parse_args()

    pairs = set(args.pairs)
    for i, pa in enumerate(pairs):
        for j, pb in enumerate(pairs):
            if j <= i:
                continue

            if pa[0] == pb[0]:
                print(f'Argument {pa[0]} is repeated')

    for p in pairs:
        print(p)


if __name__ == '__main__':
    main()
