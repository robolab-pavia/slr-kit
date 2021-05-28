import argparse
import os
import pathlib
import shutil
import sys

import toml

import scripts_defaults

SLRKIT_PATH = pathlib.Path(__file__).parent


def _check_is_dir(path):
    p = pathlib.Path(path)
    if p.is_dir():
        return p
    else:
        msg = '{!r} is not a directory'
        raise argparse.ArgumentTypeError(msg.format(path))


def init_project(args):
    meta = {
        'Project': {
            'Author': '',
            'Config': '',
            'Description': '',
            'Keywords': [],
            'Name': ''
        },
        'Source': {
            'Date': '',
            'Origin': '',
            'Query': '',
            'URL': ''
        }
    }
    old_config_dir = None
    config_dir: pathlib.Path = args.cwd / args.config_dir
    metafile = args.cwd / 'META.toml'
    if metafile.exists():
        with open(metafile) as file:
            obj = toml.load(file)

        old_config_dir = args.cwd / obj['Project']['Config']
        for k in meta:
            for h in meta[k]:
                val = obj[k].get(h, '')
                if val != '':
                    meta[k][h] = val

    if args.author:
        meta['Project']['Author'] = args.author
    if args.description:
        meta['Project']['Description'] = args.description
    if args.name:
        meta['Project']['Name'] = args.name

    directory = config_dir
    move = False
    if old_config_dir != config_dir:
        meta['Project']['Config'] = str(config_dir.name)
        if old_config_dir is not None:
            directory = old_config_dir
            move = True

    try:
        config_dir.mkdir(exist_ok=True)
    except FileExistsError:
        msg = 'Error {} exist and is not a directory'
        sys.exit(msg.format(config_dir))

    scripts = ['preprocess', 'gen-terms', 'lda', 'lda_grid_search']
    for s in scripts:
        p = (config_dir / s).with_suffix('.toml')
        old_p = (directory / s).with_suffix('.toml')
        if not old_p.exists():
            if not p.exists():
                with open(p, 'w') as file:
                    toml.dump(scripts_defaults.scripts_defaults[s], file)
        elif move:
            try:
                # create a backup copy of the destination file if exists
                shutil.copy2(p, p.with_suffix(p.suffix + '.bak'))
            except FileNotFoundError:
                # the destination file does not exist, never mind
                pass
            shutil.copy2(str(old_p), str(p))

    with open(metafile, 'w') as file:
        toml.dump(meta, file)


def check_project(args, filename):
    metafile = args.cwd / 'META.toml'
    try:
        with open(metafile) as file:
            meta = toml.load(file)
    except FileNotFoundError:
        msg = 'Error: {} is not an slr-kit project, no META.toml found'
        sys.exit(msg.format(args.cwd.resolve().absolute()))
    try:
        config_dir = args.cwd / meta['Project']['Config']
    except KeyError:
        msg = 'Error: {} is invalid invalid, no "Config" entry found'
        sys.exit(msg.format(metafile.resolve().absolute()))
    config_file = config_dir / filename
    try:
        with open(config_file) as file:
            config = toml.load(file)
    except FileNotFoundError:
        msg = 'Error: file {} not found'
        sys.exit(msg.format(config_file.resolve().absolute()))
    return config, config_dir, meta


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', type=_check_is_dir, dest='cwd',
                        default=pathlib.Path.cwd(), metavar='path',
                        help='Change directory to %(metavar)r before running '
                             'the specified command.')
    parser.add_argument('--exec-path', type=_check_is_dir, metavar='path',
                        help='Change directory to %(metavar)r before running '
                             'the specified command.')
    subparser = parser.add_subparsers(required=True)
    # init
    parser_init = subparser.add_parser('init', help='Initialize a slr-kit '
                                                    'project')
    parser_init.add_argument('config_dir', action='store', type=str,
                             default='slr-conf',
                             help='Name of the project config directory that '
                                  'will be created. If omitted %(default)r is '
                                  'used')
    parser_init.add_argument('--name', '-N', action='store', type=str,
                             default='', help='Name of the project')
    parser_init.add_argument('--author', '-A', action='store', type=str,
                             default='', help='Name of the author of the '
                                              'project')
    parser_init.add_argument('--description', '-D', action='store', type=str,
                             default='', help='Description of the project')
    parser_init.set_defaults(func=init_project)
    return parser


def main():
    args = init_argparser().parse_args()

    # execute the command
    args.func(args)


if __name__ == '__main__':
    main()
