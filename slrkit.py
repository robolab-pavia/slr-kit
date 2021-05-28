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

    scripts = ['preprocess', 'gen_terms', 'lda', 'lda_grid_search']
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


def prepare_script_arguments(config, config_dir, confname, script_name):
    args = argparse.Namespace()
    for k, v in scripts_defaults.defaults[script_name].items():
        if v.get('non-standard', False):
            continue

        dest = v.get('dest', k.replace('-', '_'))
        param = config.get(k, v['value'])
        def_val = (param == v['value'])
        if v['required']:
            if def_val:
                msg = 'Missing valid value for required parameter {!r} in {}'
                sys.exit(msg.format(k, config_dir / confname))

        if param == '' or param == []:
            setattr(args, dest, None)
        else:
            setattr(args, dest, param)
    return args


def run_preproc(args):
    script_name = 'preprocess'
    confname = '.'.join([script_name, 'toml'])
    config, config_dir, meta = check_project(args, confname)
    cmd_args = prepare_script_arguments(config, config_dir, confname, script_name)
    # handle the special parameter relevant-terms
    relterms_default = scripts_defaults.defaults[script_name]['relevant-terms']
    param = config.get('relevant-terms', relterms_default['value'])
    msg = ('parameter "relevant-terms" is not a list of list '
           'in file {}').format(config_dir / confname)
    value = None
    if param != relterms_default['value']:
        if not isinstance(param, list):
            sys.exit(msg)
        value = []
        for p in param:
            if not isinstance(p, list):
                sys.exit(msg)
            if len(p) >= 2:
                value.append(tuple(p[:2]))
            else:
                value.append((p[0], None))
    dest = relterms_default.get('dest', 'relevant_terms')
    setattr(cmd_args, dest, value)

    setattr(cmd_args, 'logfile', str(config_dir / 'slr-kit.log'))
    os.chdir(args.cwd)
    from preprocess import preprocess
    preprocess(cmd_args)


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
    # preproc
    parser_preproc = subparser.add_parser('preprocess',
                                          help='Run the preprocess stage in a '
                                               'slr-kit project')
    parser_preproc.add_argument('--config', '-c', action='store', type=str,
                                help='Alternative configuration file, in toml, '
                                     'to be used instead of the project one')
    parser_preproc.set_defaults(func=run_preproc)
    return parser


def main():
    args = init_argparser().parse_args()

    # execute the command
    args.func(args)


if __name__ == '__main__':
    main()
