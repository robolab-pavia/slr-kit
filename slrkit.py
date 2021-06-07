import argparse
import os
import pathlib
import shutil
import sys

import tomlkit

SLRKIT_PATH = pathlib.Path(__file__).parent


def _check_is_dir(path):
    p = pathlib.Path(path)
    if p.is_dir():
        return p
    else:
        msg = '{!r} is not a directory'
        raise argparse.ArgumentTypeError(msg.format(path))


def toml_load(filename):
    with open(filename) as file:
        config = tomlkit.loads(file.read())
    return config


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
    if args.config_dir is not None:
        config_dir: pathlib.Path = args.cwd / args.config_dir
    else:
        config_dir: pathlib.Path = args.cwd / f'slrkit-{args.name}'
    metafile = args.cwd / 'META.toml'
    if metafile.exists():
        obj = toml_load(metafile)

        old_config_dir = args.cwd / obj['Project']['Config']
        for k in meta:
            for h in meta[k]:
                val = obj[k].get(h, '')
                if val != '':
                    meta[k][h] = val

    meta['Project']['Name'] = args.name
    if args.author:
        meta['Project']['Author'] = args.author
    if args.description:
        meta['Project']['Description'] = args.description

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
        msg = 'Error: {} exist and is not a directory'
        sys.exit(msg.format(config_dir))

    scripts = ['preprocess', 'gen_terms', 'lda', 'lda_grid_search']
    for s in scripts:
        p = (config_dir / s).with_suffix('.toml')
        old_p = (directory / s).with_suffix('.toml')
        if not old_p.exists():
            if not p.exists():
                args: dict = __import__(s).init_argparser().slrkit_arguments
                conf = tomlkit.document()
                for arg_name, arg in args.items():
                    if not arg['log']:
                        conf.add(tomlkit.comment(arg['help']))
                        conf.add(tomlkit.comment(f'required: {arg["required"]}'))
                        if arg['value'] is not None:
                            try:
                                conf.add(arg_name, arg['value'])
                            except ValueError:
                                conf.add(arg_name, str(arg['value']))
                        else:
                            conf.add(arg_name, '')

                with open(p, 'w') as file:
                    file.write(tomlkit.dumps(conf))
        elif move:
            try:
                # create a backup copy of the destination file if exists
                shutil.copy2(p, p.with_suffix(p.suffix + '.bak'))
            except FileNotFoundError:
                # the destination file does not exist, never mind
                pass
            shutil.copy2(str(old_p), str(p))

    metadoc = tomlkit.document()
    for mk, mv in meta.items():
        tbl = tomlkit.table()
        for k, v in mv.items():
            tbl.add(k, v)

        metadoc.add(mk, tbl)
    with open(metafile, 'w') as file:
        file.write(tomlkit.dumps(metadoc))


def check_project(args, filename):
    metafile = args.cwd / 'META.toml'
    try:
        meta = toml_load(metafile)
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
        config = toml_load(config_file)
    except FileNotFoundError:
        msg = 'Error: file {} not found'
        sys.exit(msg.format(config_file.resolve().absolute()))
    return config, config_dir, meta


def prepare_script_arguments(config, config_dir, confname, script_args):
    args = argparse.Namespace()
    for k, v in script_args.items():
        if v.get('log', False):
            setattr(args, k, str((config_dir / 'slr-kit.log').resolve()))
            continue
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
    from preprocess import preprocess, init_argparser as preproc_argparse
    script_args = preproc_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    # handle the special parameter relevant-terms
    relterms_default = script_args['relevant-term']
    param = config.get('relevant-term', relterms_default['value'])
    msg = ('parameter "relevant-term" is not a list of list '
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

    os.chdir(args.cwd)
    preprocess(cmd_args)


def run_genterms(args):
    script_name = 'gen_terms'
    confname = '.'.join([script_name, 'toml'])
    config, config_dir, meta = check_project(args, confname)
    from gen_terms import gen_terms, init_argparser as gt_argparse
    script_args = gt_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    os.chdir(args.cwd)
    gen_terms(cmd_args)


def run_lda(args):
    script_name = 'lda'
    confname = '.'.join([script_name, 'toml'])
    config, config_dir, meta = check_project(args, confname)
    from lda import lda, init_argparser as lda_argparse
    script_args = lda_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    # handle the outdir parameter
    outdir_default = script_args['outdir']
    param = config.get('outdir', outdir_default['value'])
    if param != outdir_default['value']:
        setattr(cmd_args, 'outdir', (args.cwd / param).resolve())
    else:
        setattr(cmd_args, 'outdir', args.cwd.resolve())

    os.chdir(args.cwd)
    lda(cmd_args)


def run_lda_grid_search(args):
    script_name = 'lda_grid_search'
    confname = '.'.join([script_name, 'toml'])
    config, config_dir, meta = check_project(args, confname)
    from lda_grid_search import lda_grid_search, init_argparser as lda_gs_argparse
    script_args = lda_gs_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    # handle the outdir and result parameter
    outdir_default = script_args['outdir']
    param = config.get('outdir', outdir_default['value'])
    if param != outdir_default['value']:
        setattr(cmd_args, 'outdir', (args.cwd / param).resolve())
    else:
        setattr(cmd_args, 'outdir', args.cwd.resolve())

    os.chdir(args.cwd)
    result_default = script_args['result']
    param = config.get('result', result_default['value'])
    setattr(cmd_args, 'result', argparse.FileType('w')(param))

    lda_grid_search(cmd_args)


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
    subparser = parser.add_subparsers(required=True)
    # init
    parser_init = subparser.add_parser('init', help='Initialize a slr-kit '
                                                    'project')
    parser_init.add_argument('name', action='store', type=str,
                             help='Name of the project.')
    parser_init.add_argument('--config_dir', '-c', action='store', type=str,
                             help='Name of the configuration directory of the '
                                  'project. This directory will be created and '
                                  'populated with the template configuration '
                                  'files. If this directory already exists, '
                                  'only the missing template file are created. '
                                  'If this option is omitted, the directory '
                                  'will be named slrkit-<project name>.')
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
                                help='Alternative toml configuration file '
                                     'to be used instead of the project one')
    parser_preproc.set_defaults(func=run_preproc)
    # gen_terms
    parser_genterms = subparser.add_parser('gen_terms',
                                           help='Run the gen_terms stage in a '
                                                'slr-kit project')
    parser_genterms.add_argument('--config', '-c', action='store', type=str,
                                 help='Alternative toml configuration file '
                                      'to be used instead of the project one')
    parser_genterms.set_defaults(func=run_genterms)
    # lda
    parser_lda = subparser.add_parser('lda', help='Run the lda stage in a '
                                                  'slr-kit project')
    parser_lda.add_argument('--config', '-c', action='store', type=str,
                            help='Alternative toml configuration file '
                                 'to be used instead of the project one')
    parser_lda.set_defaults(func=run_lda)
    # lda_grid_search
    parser_lda_grid_search = subparser.add_parser('lda_grid_search',
                                                  help='Run the lda_grid_search'
                                                       ' stage in a slr-kit '
                                                       'project')
    parser_lda_grid_search.add_argument('--config', '-c', action='store',
                                        type=str, help='Alternative toml '
                                                       'configuration file to '
                                                       'be used instead of the '
                                                       'project one')
    parser_lda_grid_search.set_defaults(func=run_lda_grid_search)
    return parser


def main():
    args = init_argparser().parse_args()

    # execute the command
    args.func(args)


if __name__ == '__main__':
    main()
