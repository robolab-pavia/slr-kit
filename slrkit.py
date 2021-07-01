import argparse
import importlib
import os
import pathlib
import shutil
import sys

import tomlkit

SCRIPTS = {
    # config_file: module_name
    'import': {'module': 'import_biblio', 'depends': []},
    'acronyms': {'module': 'acronyms', 'depends': ['import']},
    'preprocess': {'module': 'preprocess', 'depends': ['import']},
    'gen_terms': {'module': 'gen_terms', 'depends': ['preprocess']},
    'lda': {'module': 'lda', 'depends': ['preprocess', 'gen_terms']},
    'lda_grid_search': {'module': 'lda_grid_search',
                        'depends': ['preprocess', 'gen_terms']},
    'fawoc': {'module': 'fawoc.fawoc', 'depends': ['gen_terms']},
    'report': {'module': 'topic_report', 'depends': []},
}


def _check_is_dir(path):
    p = pathlib.Path(path)
    if not p.exists():
        msg = '{!r} do not exist'
        raise argparse.ArgumentTypeError(msg.format(path))
    elif p.is_dir():
        return p
    else:
        msg = '{!r} is not a directory'
        raise argparse.ArgumentTypeError(msg.format(path))


def toml_load(filename):
    with open(filename) as file:
        config = tomlkit.loads(file.read())
    return config


def create_meta(args):
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
    config_dir: pathlib.Path = args.cwd / 'slrkit.conf'
    metafile = args.cwd / 'META.toml'
    if metafile.exists():
        obj = toml_load(metafile)

        config_dir = args.cwd / obj['Project']['Config']
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
    meta['Project']['Config'] = str(config_dir.name)
    return config_dir, meta, metafile


def prepare_configfile(modulename, metafile):
    module = importlib.import_module(modulename)
    args = module.init_argparser().slrkit_arguments
    conf = tomlkit.document()
    for arg_name, arg in args.items():
        if not arg['logfile'] and not arg['cli_only']:
            conf.add(tomlkit.comment(arg['help'].replace('\n', ' ')))
            conf.add(tomlkit.comment(f'required: {arg["required"]}'))
            if arg['suggest-suffix'] is not None:
                val = ''.join([metafile['Project']['Name'],
                               arg['suggest-suffix']])
                conf.add(arg_name, val)
            elif arg['value'] is not None:
                try:
                    conf.add(arg_name, arg['value'])
                except ValueError:
                    conf.add(arg_name, str(arg['value']))
            else:
                conf.add(arg_name, '')
    return conf, args


def init_project(args):
    config_dir, meta, metafile = create_meta(args)

    try:
        config_dir.mkdir(exist_ok=True)
        (config_dir / 'log').mkdir(exist_ok=True)
    except FileExistsError as e:
        msg = 'Error: {} exist and is not a directory'
        sys.exit(msg.format(e.filename))

    config_files = {}
    for configname, script_data in SCRIPTS.items():
        config_files[configname] = prepare_configfile(script_data['module'],
                                                      meta)

    for configname, (conf, args) in config_files.items():
        depends = SCRIPTS[configname]['depends']
        inputs = list(filter(lambda a: args[a]['input'], args))
        if len(inputs) != 0:
            for i, d in enumerate(depends):
                inp = args[inputs[i]]
                if inp['cli_only'] or inp['logfile']:
                    continue
                outputs = list(filter(lambda a: a['output'],
                                      config_files[d][1].values()))
                if len(outputs) != 1:
                    continue

                conf[inputs[i]] = ''.join([meta['Project']['Name'],
                                           outputs[0]['suggest-suffix']])

        p = (config_dir / configname).with_suffix('.toml')
        if p.exists():
            obj = toml_load(p)
            for k in conf.keys():
                conf[k] = obj.get(k, conf[k])

            shutil.copy2(p, p.with_suffix(p.suffix + '.bak'))

        with open(p, 'w') as file:
            file.write(tomlkit.dumps(conf))

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
        if v.get('non-standard', False) or v.get('cli_only', False):
            continue
        if v.get('logfile', False):
            setattr(args, k, str((config_dir / 'log' / 'slr-kit.log').resolve()))
            continue

        dest = v.get('dest', k.replace('-', '_'))
        param = config.get(k, v['value'])
        def_val = (param == v['value'] or (param == '' and v['value'] is None))
        null = param is None or param == ''
        if v['type'] is not None and not null:
            param = v['type'](param)

        if v['choices'] is not None:
            if param not in v['choices']:
                msg = 'Invalid value for parameter {!r} in {}.\nMust be one of {}'
                sys.exit(msg.format(k, config_dir / confname, v['choices']))

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
    confname = 'preprocess.toml'
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
    confname = 'gen_terms.toml'
    config, config_dir, meta = check_project(args, confname)
    from gen_terms import gen_terms, init_argparser as gt_argparse
    script_args = gt_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    os.chdir(args.cwd)
    gen_terms(cmd_args)


def run_lda(args):
    confname = 'lda.toml'
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
    confname = 'lda_grid_search.toml'
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


def run_fawoc(args):
    confname = 'fawoc.toml'
    config, config_dir, meta = check_project(args, confname)
    from fawoc.fawoc import fawoc_run, init_argparser as fawoc_argparse
    script_args = fawoc_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    # command line overrides
    if args.input is not None:
        setattr(cmd_args, 'input', args.input)
    if args.width is not None:
        setattr(cmd_args, 'width', args.width)

    # set profiler
    setattr(cmd_args, 'profiler_name',
            (config_dir / 'log' / 'fawoc_profiler.log').resolve())

    os.chdir(args.cwd)
    fawoc_run(cmd_args)


def run_import(args):
    confname = 'import.toml'
    config, config_dir, meta = check_project(args, confname)
    from import_biblio import import_data, init_argparser as import_argparse
    script_args = import_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)

    os.chdir(args.cwd)
    import_data(cmd_args)


def run_acronyms(args):
    confname = 'acronyms.toml'
    config, config_dir, meta = check_project(args, confname)
    from acronyms import acronyms, init_argparser as acro_argparse
    script_args = acro_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)

    os.chdir(args.cwd)
    acronyms(cmd_args)


def run_report(args):
    confname = 'report.toml'
    config, config_dir, meta = check_project(args, confname)
    from topic_report import report, init_argparser as report_argparse
    script_args = report_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    os.chdir(args.cwd)
    setattr(cmd_args, 'json_file', args.json_file)
    report(cmd_args)


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
    parser_init.add_argument('--author', '-A', action='store', type=str,
                             default='', help='Name of the author of the '
                                              'project')
    parser_init.add_argument('--description', '-D', action='store', type=str,
                             default='', help='Description of the project')
    parser_init.set_defaults(func=init_project)
    # import
    parser_import = subparser.add_parser('import', help='Import a bibliographic'
                                                        ' database converting '
                                                        'to the csv format used'
                                                        ' by slr-kit')

    parser_import.set_defaults(func=run_import)
    # acronyms
    parser_acronyms = subparser.add_parser('acronyms', help='Extract acronyms '
                                                            'from texts.')

    parser_acronyms.set_defaults(func=run_acronyms)
    # preproc
    parser_preproc = subparser.add_parser('preprocess',
                                          help='Run the preprocess stage in a '
                                               'slr-kit project')
    parser_preproc.set_defaults(func=run_preproc)
    # gen_terms
    parser_genterms = subparser.add_parser('gen_terms',
                                           help='Run the gen_terms stage in a '
                                                'slr-kit project')
    parser_genterms.set_defaults(func=run_genterms)
    # lda
    parser_lda = subparser.add_parser('lda', help='Run the lda stage in a '
                                                  'slr-kit project')
    parser_lda.set_defaults(func=run_lda)
    # lda_grid_search
    parser_lda_grid_search = subparser.add_parser('lda_grid_search',
                                                  help='Run the lda_grid_search'
                                                       ' stage in a slr-kit '
                                                       'project')
    parser_lda_grid_search.set_defaults(func=run_lda_grid_search)
    # fawoc
    parser_fawoc = subparser.add_parser('fawoc', help='Run fawoc in a slr-kit '
                                                      'project')
    parser_fawoc.add_argument('--input', '-i', metavar='LABEL',
                              help='Input only the terms classified with the '
                                   'specified label')
    parser_fawoc.add_argument('--width', '-w', action='store', type=int,
                              help='Width of fawoc windows.')
    parser_fawoc.set_defaults(func=run_fawoc)
    # report
    parser_report = subparser.add_parser('report', help='Run the report '
                                                        'creation script in a '
                                                        'slr-kit project')
    parser_report.add_argument('json_file', help='path to the json file '
                                                 'containing the lda '
                                                 'topic-paper results')
    parser_report.set_defaults(func=run_report)
    return parser


def main():
    args = init_argparser().parse_args()

    # execute the command
    args.func(args)


if __name__ == '__main__':
    main()
