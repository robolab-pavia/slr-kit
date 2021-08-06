import argparse
import importlib
import os
import pathlib
import shutil
import subprocess as sub
import sys

import git
import tomlkit

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


def git_add_files(files_to_commit, repo, must_exists=True):
    """
    Add files to the underlying git repository

    :param files_to_commit: list of files to commit
    :type files_to_commit: list[str]
    :param repo: repository object
    :type repo: git.repo.Repo
    :param must_exists: if True, each file in the list must exist otherwise
        an exception is raised
    :type must_exists: bool
    :raise GitError: if must_exist is True and one file does not exist
    """
    for f in files_to_commit:
        try:
            # git add the file
            # f is put in a list because add has a strange behaviour with
            # arguments
            repo.index.add([f])
        except FileNotFoundError:
            if must_exists:
                msg = None
                # clean the index
                try:
                    repo.index.reset()
                except git.exc.GitCommandError as e:
                    # this exception is expected and harmless if heads is empty
                    # (no commits yet) but if heads is not empty then something
                    # wrong happened
                    if repo.heads:
                        msg = 'Error performing the index clean up. ' \
                              'git reported: {}'.format(e)

                wd = pathlib.Path(repo.working_dir)
                err_msg = 'File {!r} not exists'.format(str(wd / f))
                if msg is not None:
                    err_msg = '{}\nAlso an another error occurred: ' \
                              '{}'.format(err_msg, msg)

                raise GitError(err_msg, '')


def git_commit(repo, commit_msg):
    """
    Commits files previously added to repository index

    :param repo: repository object
    :type repo: git.repo.Repo
    :param commit_msg: commit message
    :type commit_msg: str
    :return: True if something was added to the index and the commit is performed
    """
    # since index.commit records a commit even if no file is added, we must
    # check the index if something was really added
    # this check changes if the repository is new or if something was already
    # committed
    can_commit = False
    if not repo.heads:
        # heads is empty so nothing was committed to the repository
        # check if index.entries contains something
        if repo.index.entries:
            # something here, we can commit
            can_commit = True
    else:
        # something was committed so we can check the diff from HEAD
        if repo.index.diff('HEAD'):
            can_commit = True

    if can_commit:
        repo.index.commit(message=commit_msg)
        return True
    else:  # nothing added so nothing to commit
        return False


def init_project(slrkit_args):
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

    git_init(config_dir, metafile, slrkit_args.cwd)


def git_filelist_add_init(config_dir, metafile, wd):
    """
    Returns the list of files committed during the init of the git repository

    All paths are absolute.

    :param config_dir: path to the configuration directory
    :type config_dir: pathlib.Path
    :param metafile: path to the META.toml file
    :type metafile: pathlib.Path
    :param wd: path to the working dir of the repository
    :type wd: pathlib.Path
    :return: the list of files
    :rtype: list[str]
    """
    list_of_files = [str(f) for f in config_dir.glob('*.toml')]
    list_of_files.extend([str(metafile), str(wd / '.gitignore')])
    return list_of_files


def git_init(config_dir, metafile, cwd):
    """
    Initializes a git repository in the project and commits the first files

    It creates a .gitignore file and commits the META.toml and all the toml
    files in the configuration directory
    It ends the current process in case of error and prints a friendly error
    message

    :param config_dir: path to the configuration directory
    :type config_dir: pathlib.Path
    :param metafile: path to the META.toml
    :type metafile: pathlib.Path
    :param cwd: path to the project
    :type cwd: pathlib.Path
    :return: the repository object
    :rtype: git.repo.Repo
    """
    repo = git.repo.Repo.init(cwd)
    with open(cwd / '.gitignore', 'a') as file:
        file.write('*.json\n')
        file.write('*_lda_results\n')

    list_of_files = git_filelist_add_init(config_dir, metafile, cwd)
    try:
        git_add_files(list_of_files, repo)
    except GitError as e:
        msg = 'Error adding files to git: {!r}'.format(e.msg)
        sys.exit(msg)

    git_commit(repo, 'Project repository initialized')
    return repo


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
    """
    Prepares the arguments for a script using the content of its config file

    :param config: content of the config file
    :type config: TOMLDocument
    :param config_dir: path to the config file directory
    :type config_dir: pathlib.Path
    :param confname: name of the config file
    :type confname: str
    :param script_args: information about the script arguments
    :type script_args: dict[str, Any]
    :return: the Namespace object with the argument and the dicts of input and
        output arguments. The dict types is {arg_name: arg_value, ...}
    :rtype: tuple[argparse.Namespace, dict[str, str], dict[str, str]]
    """
    inputs = {}
    outputs = {}
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

        if v.get('input', False):
            inputs[dest] = getattr(args, dest)
        if v.get('output', False):
            outputs[dest] = getattr(args, dest)

    return args, inputs, outputs


def run_preproc(args):
    confname = 'preprocess.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from preprocess import preprocess, init_argparser as preproc_argparse
    script_args = preproc_argparse().slrkit_arguments
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
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
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
                                              script_args)
    os.chdir(args.cwd)
    script_to_run(cmd_args)


def run_lda(args):
    config_dir, meta = check_project(args.cwd)
    if args.config is not None:
        confname = pathlib.Path(args.config)
        if not confname.is_absolute():
            confname = args.cwd / confname
    else:
        confname = config_dir / 'lda.toml'
    config = load_configfile(confname)
    from lda import lda, init_argparser as lda_argparse
    script_args = lda_argparse().slrkit_arguments
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
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
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
                                              script_args)
    os.chdir(args.cwd)
    lda_ga_optimization(cmd_args)


def lda_grid_search_command(args):
    confname = 'lda_grid_search.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from lda_grid_search import lda_grid_search, init_argparser as lda_gs_argparse
    script_args = lda_gs_argparse().slrkit_arguments
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
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
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
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
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
                                              script_args)

    os.chdir(args.cwd)
    import_data(cmd_args)


def run_acronyms(args):
    confname = 'acronyms.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from acronyms import acronyms, init_argparser as acro_argparse
    script_args = acro_argparse().slrkit_arguments
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
                                              script_args)

    os.chdir(args.cwd)
    acronyms(cmd_args)


def run_report(args):
    confname = 'report.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from topic_report import report, init_argparser as report_argparse
    script_args = report_argparse().slrkit_arguments
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
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
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
                                              script_args)
    os.chdir(args.cwd)
    script_to_run(cmd_args)


def run_record(args):
    """
    Records a snapshot of the project using git

    This command uses the to_record function of the slrkit scripts. If a script
    defines a to_record function with signature:
        to_record(config: dict[str, Any]) -> list[str]
    that function is called to collect the list of files to commit.
    The to_record function is called with the dict with the content of the
    config file for the script. The to_record must return a list of string with
    the files to commit. If there is something wrong in the config file, the
    to_record must raise a ValueError. This exception is catched and its content
     is used to format the error message for the user.
    The run_record exits the current process with a friendly error message in
    case of any error.
    If the project is not a git repository, it will be git init.
    If the command is invoked with the 'clean' flag, then the index is cleaned
    from all files that are not referenced in the config files anymore.
    If the command is invoked with the 'rm' flag, then 'clean' is implied, and
    the cleaned files are also removed from the filesystem.

    :param args: cli arguments
    :type args: Namespace
    """
    if args.rm:
        args.clean = True

    config_dir, meta = check_project(args.cwd)
    metafile = args.cwd / 'META.toml'
    files = None
    if args.message == '':
        sys.exit('Error: The commit message cannot be the empty string')
    try:
        repo = git.repo.Repo(args.cwd)
    except git.exc.InvalidGitRepositoryError:
        print('Repository not yet initializated. Running git init.')
        repo = git_init(config_dir, metafile, args.cwd)
        init = True
    else:
        # try to add the files committed during init
        files = git_filelist_add_init(config_dir, metafile, args.cwd)
        init = False
        try:
            git_add_files(files, repo, must_exists=True)
        except GitError as e:
            msg = 'Error adding files to git: {!r}'.format(e.msg)
            sys.exit(msg)
    # prepare the list of files to commit
    files_to_record = []
    for k, v in SCRIPTS.items():
        mod = importlib.import_module(v['module'])
        config_file = (config_dir / k).with_suffix('.toml')
        config = load_configfile(config_file)
        if hasattr(mod, 'to_record') and callable(mod.to_record):
            try:
                files_to_record.extend(mod.to_record(config))
            except ValueError as e:
                msg = 'Error collecting files to record from {}: {}'
                sys.exit(msg.format(config_file, e.args[0]))
    wd = pathlib.Path(repo.working_dir)
    # if we have just initialized the repo, there is nothing to clean in the
    # index
    if not init and args.clean:
        # using dicts because they are faster to search
        init_files = {pathlib.Path(f): None for f in files}
        # these are the files in the index that can change
        entries = [e for e in (wd / entry for entry, _ in repo.index.entries)
                   if e not in init_files]
        # these are the files actually referred in the toml files
        files_path = {wd / f: None for f in files_to_record}
        to_clean = [entry for entry in entries if entry not in files_path]
        for f in to_clean:
            repo.index.remove(str(f), working_tree=args.rm)

    # add the fawoc profiler files
    profilers = (config_dir / 'log').glob('fawoc_*_profiler.log')
    files_to_record.extend(str(p) for p in profilers)

    git_add_files(files_to_record, repo, must_exists=False)
    if git_commit(repo, args.message):
        print('Commit correctly executed')
    else:
        print('All the file are up to date, nothing to commit')


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
                                 'the project one.')
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
    # record
    help_str = 'Record a snapshot of the project in the underlying git ' \
               'repository'
    parser_record = subparser.add_parser('record', help=help_str,
                                         description=help_str)
    parser_record.add_argument('message', help='The commit message to use for '
                                               'the commit. Cannot be the '
                                               'empty string.')
    parser_record.add_argument('--clean', action='store_true',
                               help='If set, the command cleans the repository '
                                    'index from file not referenced in the '
                                    'config files. These files are left in the '
                                    'project but they become untracked.')
    parser_record.add_argument('--rm', action='store_true',
                               help='If set, the command cleans the project '
                                    'removing files not referenced in the '
                                    'config files. This flag remove these '
                                    'files from the repository index and from '
                                    'the filesystem. Use with caution.')
    parser_record.set_defaults(func=run_record)
    # lda_grid_search
    help_str = 'Run an optimization phase for the lda stage in a ' \
               'slr-kit project using a grid search method'
    parser_lda_grid_search = subparser.add_parser('lda_grid_search',
                                                  help=help_str,
                                                  description=help_str)
    parser_lda_grid_search.set_defaults(func=lda_grid_search_command)
    return parser


def main():
    args = init_argparser().parse_args()
    # change the cwd path to absolute to be sure that everything works when
    # using paths or when changing the current dir
    args.cwd = args.cwd.absolute()
    # execute the command
    args.func(args)


if __name__ == '__main__':
    main()
