import argparse
import importlib
import os
import pathlib
import shutil
import subprocess as sub
import sys

import tomlkit


class Error(Exception):
    pass


class AddionalInitNotProvvidedError(Error):
    def __str__(self):
        return f'Additional initialization code not provvided for {self.args[0]}'


class GitError(Error):
    def __init__(self, msg, gitmsg):
        super().__init__(msg, gitmsg)
        self.msg = msg
        self.gitmsg = gitmsg


SLRKIT_DIR = pathlib.Path(__file__).parent
SCRIPTS = {
    # config_file: module_name
    'import': {'module': 'import_biblio', 'depends': [],
               'additional_init': False},
    'acronyms': {'module': 'acronyms', 'depends': ['import'],
                 'additional_init': False},
    'preprocess': {'module': 'preprocess', 'depends': ['import'],
                   'additional_init': False},
    'terms_generate': {'module': 'gen_terms', 'depends': ['preprocess'],
                       'additional_init': False},
    'lda': {'module': 'lda', 'depends': ['preprocess', 'terms_generate'],
            'additional_init': False},
    'optimize_lda': {'module': 'lda_ga',
                     'depends': ['preprocess', 'terms_generate'],
                     'additional_init': True},
    'fawoc_terms': {'module': 'fawoc.fawoc', 'depends': ['terms_generate'],
                    'additional_init': False},
    'fawoc_acronyms': {'module': 'fawoc.fawoc', 'depends': ['acronyms'],
                       'additional_init': False},
    'fawoc_journals': {'module': 'fawoc.fawoc', 'depends': ['journals_extract'],
                       'additional_init': False},
    'report': {'module': 'topic_report', 'depends': [],
               'additional_init': False},
    'journals_extract': {'module': 'journal_lister', 'depends': [],
                         'additional_init': False},
    'journals_filter': {'module': 'filter_paper',
                        'depends': ['import', 'journals_extract'],
                        'additional_init': False},
    'lda_grid_search': {'module': 'lda_grid_search',
                        'depends': ['preprocess', 'terms_generate'],
                        'additional_init': False},
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


def git_commit_files(commit_msg, files_to_commit, cwd):
    all_wrong = True
    for f in files_to_commit:
        p = sub.run(['git', 'add', f], stdout=sub.PIPE, stderr=sub.PIPE,
                    encoding='utf-8', cwd=cwd)
        if p.returncode != 0:
            print('Error: git add returned error on file', f'{f!r}',
                  'error message:', p.stdout, file=sys.stderr)
        else:
            all_wrong = False
    if all_wrong:
        msg = 'all git add operation went wrong'
        raise GitError(msg, "")
    p = sub.run(['git', 'status', '--porcelain'],
                stdout=sub.PIPE, stderr=sub.PIPE, encoding='utf-8', cwd=cwd)
    if p.returncode != 0:
        raise GitError('git status returned an error', p.stdout)
    if p.stdout != '':
        # git status returned something so we have something to commit
        p = sub.run(['git', 'commit', '-m', commit_msg],
                    stdout=sub.PIPE, stderr=sub.PIPE, encoding='utf-8', cwd=cwd)
        if p.returncode != 0:
            raise GitError('git commit returned an error', p.stdout)
    # else: all the added file are up to date and we don't have to commit them


def init_project(slrkit_args):
    # verify git presence
    try:
        sub.run(['git', '--version'], stdout=sub.DEVNULL, stderr=sub.DEVNULL)
    except FileNotFoundError:
        # no git: abort
        sys.exit('Error: git not installed')

    config_dir, meta, metafile = create_meta(slrkit_args)

    try:
        config_dir.mkdir(exist_ok=True)
        (config_dir / 'log').mkdir(exist_ok=True)
    except FileExistsError as e:
        msg = 'Error: {} exist and is not a directory'
        sys.exit(msg.format(e.filename))

    # change directory to args.cwd to be sure that all the Path.cwd() used as
    # default argument values are right
    os.chdir(slrkit_args.cwd)
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

            if not slrkit_args.no_backup:
                shutil.copy2(p, p.with_suffix(p.suffix + '.bak'))

        # the command requires extra initialization?
        if SCRIPTS[configname]['additional_init']:
            if configname == 'optimize_lda':
                ga_params = config_dir / 'optimize_lda_ga_params.toml'
                if not ga_params.exists():
                    shutil.copy2(SLRKIT_DIR / 'ga_param.toml', ga_params)

                conf['ga_params'] = str(config_dir / 'optimize_lda_ga_params.toml')
            else:  # if here, no code for the additional init
                raise AddionalInitNotProvvidedError(configname)
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

    proc = sub.run(['git', 'init'], stdout=sub.PIPE, stderr=sub.STDOUT,
                   encoding='utf-8', cwd=slrkit_args.cwd)
    if proc.returncode != 0:
        msg = 'Error: running git init: {}'
        sys.exit(msg.format(proc.stdout))

    with open(slrkit_args.cwd / '.gitignore', 'w') as file:
        file.write('*.json\n')
        file.write('*_lda_results\n')

    list_of_files = [f for f in config_dir.glob('*.toml')]
    list_of_files.extend([metafile, '.gitignore'])
    try:
        git_commit_files('Project repository initialized', list_of_files,
                         slrkit_args.cwd)
    except GitError as e:
        msg = 'Error committing files to git: {!r}'.format(e.msg)
        if e.gitmsg is not None:
            msg = '{}\n\tGit reported: {!r}'.format(msg, e.gitmsg)

        sys.exit(msg)



def check_project(cwd):
    """
    Checks if the specified path is a valid slrkit project

    It ends the current process with a friendly message in case of errors

    :param cwd: path to the directory to check
    :type cwd: pathlib.Path
    :return: the path to the configuration directory and the contents of the
        META.toml file
    :rtype: tuple[pathlib.Path, dict]
    """
    metafile = cwd / 'META.toml'
    try:
        meta = dict(toml_load(metafile))
    except FileNotFoundError:
        msg = 'Error: {} is not an slr-kit project, no META.toml found'
        sys.exit(msg.format(cwd.resolve().absolute()))
    try:
        config_dir = cwd / meta['Project']['Config']
    except KeyError:
        msg = 'Error: {} is invalid, no "Config" entry found'
        sys.exit(msg.format(metafile.resolve().absolute()))

    return config_dir, meta


def load_configfile(filename):
    config_file = pathlib.Path(filename)
    try:
        config = toml_load(config_file)
    except FileNotFoundError:
        msg = 'Error: file {} not found'
        sys.exit(msg.format(config_file.resolve().absolute()))

    return config


def prepare_script_arguments(config, config_dir, confname, script_args):
    args = argparse.Namespace()
    for k, v in script_args.items():
        if v.get('non-standard', False):
            continue
        if v.get('cli_only', False):
            setattr(args, v['dest'], v['value'])
            continue
        if v.get('logfile', False):
            setattr(args, v['dest'],
                    str((config_dir / 'log' / 'slr-kit.log').resolve()))
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
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
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


def run_terms(args):
    if args.terms_operation is None or args.terms_operation == 'generate':
        confname = 'terms_generate.toml'
        from gen_terms import (gen_terms as script_to_run,
                               init_argparser as argparser)
    else:
        msg = 'Unexpected subcommand {!r} for command terms: Aborting'
        sys.exit(msg.format(args.journals_operation))

    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    script_args = argparser().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    os.chdir(args.cwd)
    script_to_run(cmd_args)


def run_lda(args):
    if args.config is not None:
        confname = pathlib.Path(args.config)
    else:
        confname = args.cwd / 'lda.toml'

    config_dir, meta = check_project(args.cwd)
    config = load_configfile(confname)
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


def optimize_lda(args):
    confname = 'optimize_lda.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from lda_ga import lda_ga_optimization, init_argparser as lda_ga_argparse
    script_args = lda_ga_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    os.chdir(args.cwd)
    lda_ga_optimization(cmd_args)


def lda_grid_search_command(args):
    confname = 'lda_grid_search.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
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

    lda_grid_search(cmd_args)


def run_fawoc(args):
    confname = ''.join(['fawoc_', args.operation, '.toml'])
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
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
    profiler = ''.join(['fawoc_', args.operation, '_profiler.log'])
    setattr(cmd_args, 'profiler_name', (config_dir / 'log' / profiler).resolve())

    # disable the info file loading (if necessary)
    if args.operation in ['acronyms']:
        setattr(cmd_args, 'no_info_file', True)

    os.chdir(args.cwd)
    fawoc_run(cmd_args)


def run_import(args):
    confname = 'import.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from import_biblio import import_data, init_argparser as import_argparse
    script_args = import_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)

    os.chdir(args.cwd)
    import_data(cmd_args)


def run_acronyms(args):
    confname = 'acronyms.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from acronyms import acronyms, init_argparser as acro_argparse
    script_args = acro_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)

    os.chdir(args.cwd)
    acronyms(cmd_args)


def run_report(args):
    confname = 'report.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from topic_report import report, init_argparser as report_argparse
    script_args = report_argparse().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    os.chdir(args.cwd)
    setattr(cmd_args, 'json_file', args.json_file)
    report(cmd_args)


def run_journals(args):
    if args.journals_operation is None or args.journals_operation == 'extract':
        confname = 'journals_extract.toml'
        from journal_lister import (journal_lister as script_to_run,
                                    init_argparser as argparser)
    elif args.journals_operation == 'filter':
        confname = 'journals_filter.toml'
        from filter_paper import (filter_paper as script_to_run,
                                  init_argparser as argparser)
    else:
        msg = 'Unexpected subcommand {!r} for command journals: Aborting'
        sys.exit(msg.format(args.journals_operation))

    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    script_args = argparser().slrkit_arguments
    cmd_args = prepare_script_arguments(config, config_dir, confname,
                                        script_args)
    os.chdir(args.cwd)
    script_to_run(cmd_args)


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='slrkit project handling tool')
    parser.add_argument('-C', type=_check_is_dir, dest='cwd',
                        default=pathlib.Path.cwd(), metavar='path',
                        help='Change directory to %(metavar)r before running '
                             'the specified command.')
    # dest is required to avoid a crash when the user inputs no command
    subparser = parser.add_subparsers(title='slrkit commands',
                                      required=True, dest='command')
    # init
    help_str = 'Initialize a slr-kit project'
    parser_init = subparser.add_parser('init', help=help_str,
                                       description=help_str)
    parser_init.add_argument('name', action='store', type=str,
                             help='Name of the project.')
    parser_init.add_argument('--author', '-A', action='store', type=str,
                             default='', help='Name of the author of the '
                                              'project')
    parser_init.add_argument('--description', '-D', action='store', type=str,
                             default='', help='Description of the project')
    parser_init.add_argument('--no-backup', action='store_true',
                             help='Do not save the existing toml files.')
    parser_init.set_defaults(func=init_project)
    # import
    help_str = 'Import a bibliographic database converting to the csv format ' \
               'used by slr-kit.'
    parser_import = subparser.add_parser('import', help=help_str,
                                         description=help_str)

    parser_import.set_defaults(func=run_import)
    # journals
    help_str = 'Subcommand to extract and filter a list of journals. ' \
               'Requires a subcommand.'
    journals_p = subparser.add_parser('journals', help=help_str,
                                      description=help_str)

    journals_subp = journals_p.add_subparsers(title='journals commands',
                                              dest='journals_operation')
    # journal_lister
    help_str = 'Prepare a list of journals, suitable to be classified with ' \
               'fawoc.'
    journals_subp.add_parser('extract', help=help_str, description=help_str)
    # filter_paper
    help_str = 'Filters the abstracts file marking the papers published in ' \
               'the approved journals as "good".'
    journals_subp.add_parser('filter', help=help_str, description=help_str)

    journals_p.set_defaults(func=run_journals)
    # acronyms
    help_str = 'Extract acronyms from texts.'
    parser_acronyms = subparser.add_parser('acronyms', help=help_str,
                                           description=help_str)

    parser_acronyms.set_defaults(func=run_acronyms)
    # preproc
    help_str = 'Run the preprocess stage in a slr-kit project'
    parser_preproc = subparser.add_parser('preprocess', help=help_str,
                                          description=help_str)
    parser_preproc.set_defaults(func=run_preproc)
    # terms
    help_str = 'Subcommand to extract and handle lists of terms in a slr-kit ' \
               'project. Requires a sub-command'
    terms_parser = subparser.add_parser('terms', help=help_str,
                                        description=help_str)
    terms_parser.set_defaults(func=run_terms)
    terms_subp = terms_parser.add_subparsers(title='terms commands',
                                             dest='terms_operation')
    # terms_generate
    help_str = 'Generates a list of terms from documents in a slr-kit project'
    terms_subp.add_parser('generate', help=help_str, description=help_str)
    # fawoc
    help_str = 'Run fawoc in a slr-kit project.'
    parser_fawoc = subparser.add_parser('fawoc', help=help_str,
                                        description=help_str)
    parser_fawoc.add_argument('operation', default='terms', nargs='?',
                              choices=['terms', 'acronyms', 'journals'],
                              help='Specifies what the user wants to '
                                   'classify with fawoc. This argument can be '
                                   'one of %(choices)r. '
                                   'Default: %(default)r.')
    parser_fawoc.add_argument('--input', '-i', metavar='LABEL',
                              help='Input only the terms classified with the '
                                   'specified label')
    parser_fawoc.add_argument('--width', '-w', action='store', type=int,
                              help='Width of fawoc windows.')
    parser_fawoc.set_defaults(func=run_fawoc)
    # lda
    help_str = 'Run the lda stage in a slr-kit project'
    parser_lda = subparser.add_parser('lda', help=help_str,
                                      description=help_str)
    parser_lda.add_argument('--config', '-c',
                            help='Path to the toml file to be used instead of '
                                 'the project one')
    parser_lda.set_defaults(func=run_lda)
    # report
    help_str = 'Run the report creation script in a slr-kit project.'
    parser_report = subparser.add_parser('report', help=help_str,
                                         description=help_str)
    parser_report.add_argument('json_file', help='Path to the json file '
                                                 'containing the LDA '
                                                 'topic-paper results.')
    parser_report.set_defaults(func=run_report)
    # optimize_lda
    help_str = 'Run an optimization phase for the lda stage in a' \
               'slr-kit project, using a GA.'
    parser_optimize_lda = subparser.add_parser('optimize_lda',
                                               help=help_str,
                                               description=help_str)
    parser_optimize_lda.set_defaults(func=optimize_lda)
    # lda_grid_search
    help_str = 'Run an optimization phase for the lda stage in a ' \
               'slr-kit project using a grid search method'
    parser_optimize_lda = subparser.add_parser('lda_grid_search',
                                               help=help_str,
                                               description=help_str)
    parser_optimize_lda.set_defaults(func=lda_grid_search_command)
    return parser


def main():
    args = init_argparser().parse_args()
    # execute the command
    args.func(args)


if __name__ == '__main__':
    main()
