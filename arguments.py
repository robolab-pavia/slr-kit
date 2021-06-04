import abc
import argparse


class SlrKitAction(argparse.Action, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def slrkit_conf_default(self):
        raise NotImplementedError


class AppendMultipleFilesAction(SlrKitAction):
    """
    Action for argparse that collects multiple option arguments as a set

    This can be used to implement a option that can have multiple arguments.
    The option itself may be given multiple time on the command line.
    All the arguments are collected in a set of string.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            if ((isinstance(nargs, str) and nargs in ['*', '?'])
                    or (isinstance(nargs, int) and nargs < 0)):
                raise ValueError(f'nargs = {nargs} is not allowed')

        super().__init__(option_strings, dest, nargs, **kwargs)

    @property
    def slrkit_conf_default(self):
        return []

    def __call__(self, parser, namespace, values, option_string=None):
        files = getattr(namespace, self.dest, None)
        if files is None:
            files = set()

        if not isinstance(values, list):
            values = [values]

        for v in values:
            files.add(v)

        setattr(namespace, self.dest, files)


class AppendMultiplePairsAction(SlrKitAction):
    """
    Action for argparse that collects multiple option pair of arguments

    This can be used to implement a option that can have at most 2 arguments.
    The option itself may be given multiple time on the command line.
    Each pair is memorized as a tuple of string.
    All pairs are collected in a list of tuples.
    The unique_first argument specifies if the first element of each pair must
    be unique.
    The unique_second argument specifies if the second element of each pair must
    be unique.
    """

    def __init__(self, option_strings, dest, nargs=None, unique_first=False,
                 unique_second=False, **kwargs):
        if nargs is not None:
            if ((isinstance(nargs, str) and nargs in ['*', '?'])
                    or (isinstance(nargs, int) and nargs < 2)):
                raise ValueError(f'nargs = {nargs} is not allowed')

        self._unique_first = unique_first
        self._unique_second = unique_second
        self._tocheck = []
        if self._unique_first:
            self._tocheck.append((0, 'first'))
        if self._unique_second:
            self._tocheck.append((1, 'second'))

        super().__init__(option_strings, dest, nargs, **kwargs)

    @property
    def slrkit_conf_default(self):
        return [[]]

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

        for index, pos in self._tocheck:
            if any(p[index] == pair[index] for p in pairs):
                msg = f'The {pos} element of argument pair {len(pairs) + 1} ' \
                      'is not unique'
                raise argparse.ArgumentError(self, message=msg)

        pairs.append(pair)

        setattr(namespace, self.dest, pairs)
