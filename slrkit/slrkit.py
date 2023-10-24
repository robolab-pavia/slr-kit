import argparse
import datetime
import importlib
import os
import pathlib
import shutil
import sys

import git
import tomlkit
import pandas as pd
from slrkit_utils.argument_parser import ArgParse

from utils import assert_column
from version import __slrkit_version__

SLRKIT_DIR = pathlib.Path(__file__).parent
STOPWORDS_DIR = SLRKIT_DIR / 'default_stopwords'
SCRIPTS = {
    'import': {'module': 'import_biblio', 'depends': [],
               'additional_init': False, 'no_config': False},
    'journals_extract': {'module': 'journal_lister', 'depends': ['import'],
                         'additional_init': False, 'no_config': False},
    'journals_filter': {'module': 'filter_paper',
                        'depends': ['import', 'journals_extract'],
                        'additional_init': False, 'no_config': False},
    'acronyms': {'module': 'acronyms', 'depends': ['import'],
                 'additional_init': False, 'no_config': False},
    'preprocess': {'module': 'preprocess', 'depends': ['import'],
                   'additional_init': True, 'no_config': False},
    'terms_generate': {'module': 'gen_terms', 'depends': ['preprocess'],
                       'additional_init': False, 'no_config': False},
    'fawoc_terms': {'module': 'fawoc.fawoc', 'depends': ['terms_generate'],
                    'additional_init': False, 'no_config': False},
    'fawoc_acronyms': {'module': 'fawoc.fawoc', 'depends': ['acronyms'],
                       'additional_init': False, 'no_config': False},
    'fawoc_journals': {'module': 'fawoc.fawoc', 'depends': ['journals_extract'],
                       'additional_init': False, 'no_config': False},
    'postprocess': {'module': 'postprocess',
                    'depends': ['preprocess', 'terms_generate'],
                    'additional_init': False, 'no_config': False},
    'lda': {'module': 'lda', 'depends': ['postprocess'],
            'additional_init': False, 'no_config': False},
    'report': {'module': 'topic_report', 'depends': ['import'],
               'additional_init': False, 'no_config': False},
    'optimize_lda': {'module': 'lda_ga',
                     'depends': ['postprocess'],
                     'additional_init': True, 'no_config': False},
    'stopwords': {'module': 'stopword_extractor',
                  'depends': ['terms_generate'],
                  'additional_init': False, 'no_config': True},
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


class ArgParseActionError(Error):
    pass


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


def prepare_configfile(modulename, metafile, project_dir):
    module = importlib.import_module(modulename, 'slrkit')
    # noinspection PyUnresolvedReferences
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
                    if isinstance(arg['value'], pathlib.Path):
                        conf.add(arg_name,
                                 str(arg['value'].relative_to(project_dir)))
                    else:
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


def run_init(slrkit_args):
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
        if script_data['no_config']:
            continue
        config_files[configname] = prepare_configfile(script_data['module'],
                                                      meta, slrkit_args.cwd)
    ignore_list = []
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
        mod = importlib.import_module(SCRIPTS[configname]['module'], 'slrkit')
        if hasattr(mod, 'to_ignore') and callable(mod.to_ignore):
            try:
                ignore_list.extend(mod.to_ignore(conf))
            except ValueError:
                pass  # at this stage the config files can be incomplete

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

                conf['ga_params'] = str(ga_params.relative_to(slrkit_args.cwd))
            elif configname == 'preprocess':
                stopwords = STOPWORDS_DIR.glob('*')
                for f in stopwords:
                    sw = pathlib.Path.cwd() / f.name
                    if not sw.exists():
                        shutil.copy2(f, sw)
                    conf['stop-words'].append(f.name)
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

    git_init(config_dir, metafile, slrkit_args.cwd, ignore_list)


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


def git_init(config_dir, metafile, cwd, ignore_list):
    """
    Initializes a git repository in the project and commits the first files

    It creates a .gitignore file based on ignore_list and commits the META.toml
    and all the toml files in the configuration directory.
    It ends the current process in case of error and prints a friendly error
    message

    :param config_dir: path to the configuration directory
    :type config_dir: pathlib.Path
    :param metafile: path to the META.toml
    :type metafile: pathlib.Path
    :param cwd: path to the project
    :type cwd: pathlib.Path
    :param ignore_list: list of file names to insert in .gitignore
    :type ignore_list: list[str]
    :return: the repository object
    :rtype: git.repo.Repo
    """
    repo = git.repo.Repo.init(cwd)
    ignore_list = set(ignore_list)
    log = config_dir / 'log' / 'slr-kit.log'
    ignore_list.add(str(log.relative_to(cwd)))
    ignore_list.add('*fawoc_data.json')

    with open(cwd / '.gitignore', 'a') as file:
        file.write('\n'.join(ignore_list))

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


def _argparse_exit(_=0, message=None):
    raise ArgParseActionError(message)


def _argparse_error(message):
    _argparse_exit(message=message)


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
    parser = ArgParse()
    parser.exit = _argparse_exit
    parser.error = _argparse_error
    args = argparse.Namespace()
    for k, v in script_args.items():
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
        default = v['action'].default
        null = param is None or param == ''
        if v['type'] is not None and isinstance(param, str) and not null:
            param = v['type'](param)

        if v['choices'] is not None:
            if param not in v['choices']:
                msg = 'Invalid value for parameter {!r} in {}.\nMust be one of {}'
                sys.exit(msg.format(k, config_dir / confname, v['choices']))

        if v['required']:
            if def_val:
                msg = 'Missing valid value for required parameter {!r} in {}'
                sys.exit(msg.format(k, config_dir / confname))

        if not def_val:
            try:
                if not isinstance(param, list):
                    param = [param]
                for p in param:
                    v['action'](parser, args, p)

            except ArgParseActionError as e:
                msg = 'Invalid value for parameter {!r} in {}.\n{}'
                sys.exit(msg.format(k, config_dir / confname, e.args[0]))
        else:
            if v['type'] is not None and isinstance(default, str):
                default = v['type'](default)

            setattr(args, dest, default)

        if v.get('input', False):
            inputs[dest] = getattr(args, dest)
        if v.get('output', False):
            outputs[dest] = getattr(args, dest)

    return args, inputs, outputs


def check_dependencies(inputs, commandname, cwd):
    """
    Checks if the inputs are present and suggest the command to run if they are missing

    This function returns error messages for each input that is missing
    suggesting the command to run to create it.
    If all the inputs exists, it returns an empty list

    :param inputs: dict of the inputs as returned by prepare_script_arguments
    :type inputs: dict[str, Any]
    :param commandname: name of the invoked command
    :type commandname: str
    :param cwd: current directory (arg of the -C option)
    :type cwd: pathlib.Path
    :return: a list of messages to print to the user if one or more inputs does
        not exists or an empty list if all inputs exists
    :rtype: list[str]
    """
    msgs = []
    for i, (k, v) in enumerate(inputs.items()):
        p = pathlib.Path(v)
        if not p.is_absolute():
            p = cwd / p

        if not p.exists():
            dep = SCRIPTS[commandname]['depends'][i]
            dep = ' '.join(dep.split('_'))
            msg = 'Error: input file {!r} missing. Run the {!r} command to ' \
                  'create it'
            msgs.append(msg.format(str(p), dep))

    return msgs


def run_preproc(args):
    confname = 'preprocess.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from preprocess import preprocess, init_argparser as preproc_argparse
    script_args = preproc_argparse().slrkit_arguments
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir, confname,
                                                   script_args)
    msgs = check_dependencies(inputs, 'preprocess', args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)

    os.chdir(args.cwd)
    preprocess(cmd_args)


def run_postprocess(args):
    confname = 'postprocess.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from postprocess import postprocess, init_argparser as postproc_argparse
    script_args = postproc_argparse().slrkit_arguments
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir,
                                                   confname, script_args)
    msgs = check_dependencies(inputs, 'postprocess', args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)

    os.chdir(args.cwd)
    postprocess(cmd_args)


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
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir, confname,
                                                   script_args)
    msgs = check_dependencies(inputs, 'terms_generate', args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)

    os.chdir(args.cwd)
    script_to_run(cmd_args)


def run_topics(args):
    if args.topics is None:
        print('Error: missing sub-command name for command "topics"')
        sys.exit(1)
    elif args.topics == 'extract':
        run_lda(args)
    elif args.topics == 'optimize':
        run_optimize_lda(args)
    else:
        msg = 'Error: unknown sub-command {!r} for command "topics"'
        print(msg.format(args.topics))
        sys.exit(1)


def run_lda(args):
    config_dir, meta = check_project(args.cwd)
    if args.directory is None and (args.uuid is not None or args.id is not None):
        msg = 'Error: missing --directory option. It is required by the {} ' \
              'option'
        if args.uuid is not None:
            msg = msg.format('--uuid')
        else:
            msg = msg.format('--id')
        print(msg)
        sys.exit(1)
    if args.id is None:
        args.id = 0
    if args.config is not None:
        confname = pathlib.Path(args.config)
        if not confname.is_absolute():
            confname = args.cwd / confname
    elif args.directory is not None:
        confname = pathlib.Path(args.directory)
        if not confname.is_absolute():
            confname = args.cwd / confname

        if args.uuid is not None:
            confname = (confname / 'toml' / args.uuid).with_suffix('.toml')
        else:
            resfile = confname / 'results.csv'
            try:
                df = pd.read_csv(resfile, delimiter='\t',
                                 index_col='id')
            except FileNotFoundError as err:
                msg = 'Error: file {!r} not found'
                sys.exit(msg.format(err.filename))
            except ValueError:
                msg = 'File {!r} must contain the {!r} column.'
                sys.exit(msg.format(str(resfile), 'id'))

            assert_column(str(resfile), df, ['uuid'])
            try:
                uid = df.at[args.id, 'uuid']
            except KeyError:
                msg = 'Error: id {!r} not valid in the {!r} result directory'
                sys.exit(msg.format(args.id, str(resfile)))

            del df
            print('Loading model with id', args.id, 'and uuid', uid)
            confname = (confname / 'toml' / uid).with_suffix('.toml')
    else:
        confname = config_dir / 'lda.toml'

    os.chdir(args.cwd)
    config = load_configfile(confname)
    from lda import lda, init_argparser as lda_argparse
    script_args = lda_argparse().slrkit_arguments
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir, confname,
                                                   script_args)
    msgs = check_dependencies(inputs, 'lda', args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)

    # check the seed parameter: if it is set to '' (no seed set) change it to
    # None. In this way the lda code will work
    if cmd_args.seed == '':
        cmd_args.seed = None

    # this is required to set the PYTHONHASHSEED variable from our code and
    # ensure the reproducibility of the lda train
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    os.putenv('PYTHONHASHSEED', '0')
    p = multiprocessing.Process(target=lda, args=(cmd_args, ))
    p.start()
    p.join()


def run_optimize_lda(args):
    confname = 'optimize_lda.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from lda_ga import lda_ga_optimization, init_argparser as lda_ga_argparse
    script_args = lda_ga_argparse().slrkit_arguments
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir, confname,
                                                   script_args)
    msgs = check_dependencies(inputs, 'optimize_lda', args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)

    os.chdir(args.cwd)

    # check the seed parameter: if it is set to '' (no seed set) change it to
    # None. In this way the lda code will work
    if cmd_args.seed == '':
        cmd_args.seed = None

    # this is required to set the PYTHONHASHSEED variable from our code and
    # ensure the reproducibility of the lda train
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    os.putenv('PYTHONHASHSEED', '0')
    p = multiprocessing.Process(target=lda_ga_optimization, args=(cmd_args, ))
    p.start()
    p.join()


def run_fawoc(args):
    if args.fawoc_operation is None:
        args.fawoc_operation = 'terms'

    command_name = '_'.join(['fawoc', args.fawoc_operation])
    confname = ''.join([command_name, '.toml'])
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from fawoc.fawoc import fawoc_run, init_argparser as fawoc_argparse
    script_args = fawoc_argparse().slrkit_arguments
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir, confname,
                                                   script_args)
    msgs = check_dependencies(inputs, command_name, args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)
    # command line overrides
    if args.input is not None:
        setattr(cmd_args, 'input', args.input)
    if args.width is not None:
        setattr(cmd_args, 'width', args.width)

    # set profiler
    profiler = ''.join(['fawoc_', args.fawoc_operation, '_profiler.log'])
    setattr(cmd_args, 'profiler_name', (config_dir / 'log' / profiler).resolve())

    # disable the info file loading (if necessary)
    if args.fawoc_operation in ['acronyms']:
        setattr(cmd_args, 'no_info_file', True)

    os.chdir(args.cwd)
    fawoc_run(cmd_args)


def run_import(args):
    confname = 'import.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from import_biblio import import_data, init_argparser as import_argparse
    script_args = import_argparse().slrkit_arguments
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir, confname,
                                                   script_args)
    if args.list_columns:
        cmd_args.columns = '?'

    os.chdir(args.cwd)
    import_data(cmd_args)


def run_acronyms(args):
    confname = 'acronyms.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from acronyms import acronyms, init_argparser as acro_argparse
    script_args = acro_argparse().slrkit_arguments
    cmd_args, inputs, outputs = prepare_script_arguments(config, config_dir, confname,
                                                         script_args)
    msgs = check_dependencies(inputs, 'acronyms', args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)

    os.chdir(args.cwd)
    acronyms(cmd_args)
    preproc_config = load_configfile(config_dir / 'preprocess.toml')
    # change the preprocess.toml file adding the acronyms output in the acronyms
    # field
    preproc_config['acronyms'] = outputs['output']
    # save the new file
    with open(config_dir / 'preprocess.toml', 'w') as file:
        file.write(tomlkit.dumps(preproc_config))


def run_report(args):
    confname = 'report.toml'
    config_dir, meta = check_project(args.cwd)
    config = load_configfile(config_dir / confname)
    from topic_report import report, init_argparser as report_argparse
    script_args = report_argparse().slrkit_arguments
    cmd_args, _, _ = prepare_script_arguments(config, config_dir, confname,
                                              script_args)
    os.chdir(args.cwd)
    if args.lda_results_path is None:
        path = args.cwd
    else:
        path = args.lda_results_path
        if not path.exists():
            print('Error: specified lda_results_path not found')
            sys.exit(1)

    # get the list of the docs topics association file created with lda
    # this list is sorted by modification time, so results_list[0] is the
    # most recent file
    results_list = sorted(path.glob('lda_docs-topics*.json'),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if not results_list:
        msg = "Error: No lda_docs-topics json file found in {!r}. " \
              "Run the 'lda' command first."
        sys.exit(msg.format(str(path)))

    docs_topics_file = str(results_list[0])
    import re
    m = re.search(r'lda_docs-topics(?P<timestamp>_[-0-9_]+)?'
                  r'(?P<uuid>_[-0-9a-f]{36})?\.json',
                  docs_topics_file)
    grp = m.groupdict()
    if grp['timestamp'] is None:
        grp['timestamp'] = ''
    if grp['uuid'] is None:
        grp['uuid'] = ''
    terms_topics_file = ''.join(['lda_terms-topics', grp['timestamp'],
                                 grp['uuid'], '.json'])

    setattr(cmd_args, 'docs_topics_file', str(docs_topics_file))
    setattr(cmd_args, 'terms_topics_file', str(path / terms_topics_file))
    timestamp = f'{datetime.datetime.now():%Y-%m-%d_%H%M%S}'
    dirpath = args.cwd / ''.join([timestamp, grp['uuid'], '_report'])
    setattr(cmd_args, 'dir', str(dirpath))

    report(cmd_args)


def run_journals(args):
    """
    Run the journals command

    The journals_operation attribute of args must contains the name of the
    sub-command to run (extract or filter).
    Usually this function checks for the dependencies and if one or more input
    files are missing it prints the corresponding error messages produced by
    check_dependencies and calls sys.exit(1).
    If args.command is 'build', this means that the function is called as part
    of the build command. In this case if the check for the dependencies fails,
    it prints a warning message and return False
    If every checks are ok and the execution does not give any error this
    function returns True.
    :param args: the arguments from the command line
    :type args: argparse.Namespace
    :return: True if everything ok. False if the dependenies check fails and the
        the function is run from the build command
    :rtype: bool
    """
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
    cmd_args, inputs, _ = prepare_script_arguments(config, config_dir, confname,
                                                   script_args)
    msgs = check_dependencies(inputs, 'journals_filter', args.cwd)
    if msgs:
        if args.command == 'build':
            # we are called by the build command and no journals file found
            # print a warning and return
            print('No journals file found - SKIP')
            return False
        for m in msgs:
            print(m)
        if args.journals_operation == 'filter':
            print("Remember to run the 'fawoc journals' command to classify",
                  "the list of journals")
        sys.exit(1)

    os.chdir(args.cwd)
    script_to_run(cmd_args)
    return True


def run_stopwords(args):
    config_dir, meta = check_project(args.cwd)
    confname = 'terms_generate.toml'
    config = load_configfile(config_dir / confname)
    from stopword_extractor import (stopword_extractor,
                                     init_argparser as argparser)
    from gen_terms import init_argparser as gen_terms_argparse
    script_args = argparser().slrkit_arguments
    gen_terms_args = gen_terms_argparse().slrkit_arguments
    del gen_terms_argparse
    input_ = None
    output = None
    for arg in script_args.values():
        if arg['input']:
            input_ = arg['dest']
        if arg['output']:
            output = arg['dest']
    terms_file = None
    for name, arg in gen_terms_args.items():
        if arg['output']:
            terms_file = name
    terms_file = config[terms_file]
    del gen_terms_args, config
    cmd_args = argparse.Namespace()
    setattr(cmd_args, input_, terms_file)
    setattr(cmd_args, output, args.output)
    msgs = check_dependencies({input_: terms_file}, 'stopwords', args.cwd)
    if msgs:
        for m in msgs:
            print(m)
        sys.exit(1)

    os.chdir(args.cwd)
    stopword_extractor(cmd_args)
    if not args.no_add:
        conf_file = config_dir / 'preprocess.toml'
        config = load_configfile(conf_file)
        config['stop-words'].append(args.output)
        # save the new file
        with open(conf_file, 'w') as file:
            file.write(tomlkit.dumps(config))


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
    repo = None
    if args.message == '':
        sys.exit('Error: The commit message cannot be the empty string')
    try:
        repo = git.repo.Repo(args.cwd)
    except git.exc.InvalidGitRepositoryError:
        print('Repository not yet initializated. Running git init.')
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
    # prepare the list of files to commit and to ignore
    files_to_record = []
    ignore_list = []
    for k, v in SCRIPTS.items():
        if v['no_config']:
            continue
        mod = importlib.import_module(v['module'], 'slrkit')
        config_file = (config_dir / k).with_suffix('.toml')
        config = load_configfile(config_file)
        if hasattr(mod, 'to_record') and callable(mod.to_record):
            try:
                files_to_record.extend(mod.to_record(config))
            except ValueError as e:
                msg = 'Error collecting files to record from {}: {}'
                sys.exit(msg.format(config_file, e.args[0]))
        if init and hasattr(mod, 'to_ignore') and callable(mod.to_ignore):
            try:
                ignore_list.extend(mod.to_ignore(config))
            except ValueError as e:
                msg = 'Error collecting files to ignore from {}: {}'
                sys.exit(msg.format(config_file, e.args[0]))

    if init:
        repo = git_init(config_dir, metafile, args.cwd, ignore_list)

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
    for p in profilers:
        files_to_record.append(str(p))
        try:
            with open(p) as file:
                content = file.read()
        except FileNotFoundError:
            # no file, so no substitutions to make
            continue
        else:
            import re
            content = re.sub(str(wd), 'PROJECT_DIR', content)
            with open(p, 'w') as file:
                file.write(content)

    # who knows why the '.' directory is added to the list
    # this causes the commit of the .git directory, which is bad
    files_to_record = [x for x in files_to_record if x != '.']
    files_to_record.append('README.md')
    git_add_files(files_to_record, repo, must_exists=False)
    if git_commit(repo, args.message):
        print('Commit correctly executed')
    else:
        print('All the file are up to date, nothing to commit')


def run_build(args):
    """
    Runs the commands required to rebuild a project after git clone

    It runs:
    - import
    - journals filter
    - preprocess

    :param args: arguments from the command line
    :type args: argparse.Namespace
    """
    # run_import requires the list_columns arg that is a boolean
    setattr(args, 'list_columns', False)
    print('Running import')
    run_import(args)
    print('import: DONE')
    # run_journals requires the journals_operation arg to determine which
    # sub-command to run. In this case we want the journals filter
    setattr(args, 'journals_operation', 'filter')
    print('Running journals filter')
    if run_journals(args):
        print('journals filters: DONE')

    print('Running preprocess')
    run_preproc(args)
    print('preprocess: DONE')


def run_readme(args):
    config_dir, meta = check_project(args.cwd)
    with open('README.md', 'w') as file:
        print(f'# {meta["Project"]["Name"]}', file=file)
        print(f'{meta["Project"]["Description"]}', file=file)
        print(file=file)
        print(f'Author: {meta["Project"]["Author"]}', file=file)
        print(file=file)
        if meta['Source']['Date']:
            print(f'Documents retrieved on: {meta["Source"]["Date"]}', file=file)
            print(file=file)
        if meta['Source']['URL'] != '' or meta['Source']['Source'] != '':
            print('## Source of the documents', file=file)
            if meta['Source']['URL'] != '':
                print(f'{meta["Source"]["URL"]}', file=file)
            else:
                print(f'{meta["Source"]["Source"]}', file=file)
            print(file=file)
            if meta['Source']['Query']:
                print('### Query used to retrieve the documents', file=file)
                print(file=file)
                print(f'    {meta["Source"]["Query"]}', file=file)
                print(file=file)
    try:
        repo = git.repo.Repo(args.cwd)
    except git.exc.InvalidGitRepositoryError:
        pass
    else:
        git_add_files(['README.md'], repo, must_exists=False)
        git_commit(repo, 'Add autogenerated README file')


def subparser_build(subparser):
    help_str = 'Run all the commands needed to re-create all the files after ' \
               'cloning a slr-kit project.'
    parser_build = subparser.add_parser('build', help=help_str,
                                        description=help_str)
    parser_build.set_defaults(func=run_build)


def subparser_readme(subparser):
    help_str = 'Creates a README.md file for the project using information ' \
               'from the META.toml file. The README.md is also committed to ' \
               'the git repository.'
    parser_readme = subparser.add_parser('readme', help=help_str,
                                         description=help_str)
    parser_readme.set_defaults(func=run_readme)


def subparser_record(subparser):
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


def subparser_topics_optimize(subparser):
    help_str = 'Run an optimization phase for the topics extraction stage in ' \
               'a slr-kit project, using a GA.'
    parser_optimize_lda = subparser.add_parser('optimize',
                                               help=help_str,
                                               description=help_str)
    parser_optimize_lda.set_defaults()


def subparser_report(subparser):
    help_str = 'Run the report creation script in a slr-kit project.'
    descr = 'Without arguments it takes the last lda results found in the ' \
            'project. If the user specifies a path as an argument, that ' \
            'directory is searched for lda results to use (always the last ' \
            'in cronological order).'
    parser_report = subparser.add_parser('report', help=help_str,
                                         description=' '.join([help_str, descr]))
    parser_report.add_argument('lda_results_path', nargs='?',
                               help='Path to the directory where the files '
                                    'containing the LDA results to use are '
                                    'stored.', type=pathlib.Path)
    parser_report.set_defaults(func=run_report)


def subparser_topics(subparser):
    help_str = 'Subcommand to extract the topics from the documents in a ' \
               'slrkit project. This command requires a subcommand.'
    parser_lda = subparser.add_parser('topics', help=help_str,
                                      description=help_str)
    topics_subp = parser_lda.add_subparsers(title='topics commands',
                                            dest='topics')

    subparser_topics_extract(topics_subp)
    subparser_topics_optimize(topics_subp)
    parser_lda.set_defaults(func=run_topics)


def subparser_topics_extract(topics_subp):
    help_str = 'Extract topics from the documents in a slr-kit project'
    extract_parser = topics_subp.add_parser('extract', help=help_str,
                                            description=help_str)
    group = extract_parser.add_mutually_exclusive_group()
    group.add_argument('--config', '-c',
                       help='Path to the toml file to be used instead of the '
                            'project one.')
    group.add_argument('--directory', '-d',
                       help='Path to the directory with the results of the '
                            'optimization phase.')
    group2 = extract_parser.add_mutually_exclusive_group()
    group2.add_argument('--uuid', '-u',
                        help='UUID of the model stored in the result directory. '
                             'If this option is given without the --directory '
                             'option, the command ends with an error.')
    group2.add_argument('--id', type=int,
                        help='0-based id of the model stored in the result '
                             'directory. The associaction between id and model '
                             'is stored in the `results.csv` file of the '
                             'result directory. This file is sorted by '
                             'coherence so the id 0 is the best model. If this '
                             'option is given without the --directory option, '
                             'the command ends with an error. If both --uuid '
                             'and this option are missing and the --directory '
                             'is present, --id is assumed with value '
                             '%(default)r')


def subparser_fawoc(subparser):
    help_str = 'Run fawoc in a slr-kit project. This command accepts a ' \
               'subcommand. If none is given "terms" is assumed.'
    parser_fawoc = subparser.add_parser('fawoc', help=help_str,
                                        description=help_str)
    parser_fawoc.add_argument('--input', '-i', metavar='LABEL',
                              help='Input only the terms classified with the '
                                   'specified label')
    parser_fawoc.add_argument('--width', '-w', action='store', type=int,
                              help='Width of fawoc windows.')
    fawoc_subp = parser_fawoc.add_subparsers(title='fawoc commands',
                                             dest='fawoc_operation')
    # fawoc_terms
    help_str = 'Classifies the terms list. This is the default command if ' \
               'none is given.'
    fawoc_subp.add_parser('terms', help=help_str, description=help_str)
    # fawoc_journals
    help_str = 'Classifies the journals list.'
    fawoc_subp.add_parser('journals', help=help_str, description=help_str)
    # fawoc_acronyms
    help_str = 'Classifies the acronyms list.'
    fawoc_subp.add_parser('acronyms', help=help_str, description=help_str)
    parser_fawoc.set_defaults(func=run_fawoc)


def subparser_terms(subparser):
    help_str = 'Subcommand to extract and handle lists of terms in a slr-kit ' \
               'project. This command accepts a subcommand. If none is ' \
               'given, "generate" is assumed.'
    terms_parser = subparser.add_parser('terms', help=help_str,
                                        description=help_str)
    terms_parser.set_defaults(func=run_terms)
    terms_subp = terms_parser.add_subparsers(title='terms commands',
                                             dest='terms_operation')
    # terms_generate
    help_str = 'Generates a list of terms from documents in a slr-kit project.' \
               ' This is the default command if none is given.'
    terms_subp.add_parser('generate', help=help_str, description=help_str)


def subparser_preproc(subparser):
    help_str = 'Run the preprocess stage in a slr-kit project'
    parser_preproc = subparser.add_parser('preprocess', help=help_str,
                                          description=help_str)
    parser_preproc.set_defaults(func=run_preproc)


def subparser_postproc(subparser):
    help_str = 'Run the postprocess stage in a slr-kit project'
    parser_postproc = subparser.add_parser('postprocess', help=help_str,
                                           description=help_str)
    parser_postproc.set_defaults(func=run_postprocess)


def subparser_acronyms(subparser):
    help_str = 'Extract acronyms from texts.'
    parser_acronyms = subparser.add_parser('acronyms', help=help_str,
                                           description=help_str)
    parser_acronyms.set_defaults(func=run_acronyms)


def subparser_journals(subparser):
    help_str = 'Subcommand to extract and filter a list of journals. ' \
               'This command accepts a subcommand. If none is given, ' \
               '"extract" is assumed.'
    journals_p = subparser.add_parser('journals', help=help_str,
                                      description=help_str)
    journals_subp = journals_p.add_subparsers(title='journals commands',
                                              dest='journals_operation')
    # journal_lister
    help_str = 'Prepare a list of journals, suitable to be classified with ' \
               'fawoc. This is the default sub-command if none is given.'
    journals_subp.add_parser('extract', help=help_str, description=help_str)
    # filter_paper
    help_str = 'Filters the abstracts file marking the papers published in ' \
               'the approved journals as "good".'
    journals_subp.add_parser('filter', help=help_str, description=help_str)
    journals_p.set_defaults(func=run_journals)


def subparser_import(subparser):
    help_str = 'Import a bibliographic database converting to the csv format ' \
               'used by slr-kit.'
    parser_import = subparser.add_parser('import', help=help_str,
                                         description=help_str)
    parser_import.add_argument('--list_columns', action='store_true',
                               help='If set, the command outputs only the list '
                                    'of available columns in the input_file '
                                    'specified in the configuration file. '
                                    'No other operation are performed and no '
                                    'data is imported.')
    parser_import.set_defaults(func=run_import)


def subparser_init(subparser):
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
    parser_init.set_defaults(func=run_init)


def subparser_stopword(subparser):
    help_str = 'Extracts the terms classified as stopwords from the terms file'
    parser_init = subparser.add_parser('stopwords', help=help_str,
                                       description=help_str)
    parser_init.add_argument('output', action='store', type=str,
                             help='File where to store the stopwords. This file'
                                  ' will be added to the "stop-words" list in '
                                  '"preprocess.toml"')
    parser_init.add_argument('--no-add', action='store_true',
                             help='Do not add the output file to the '
                                  '"stop-words" list in "preprocess.toml"')
    parser_init.set_defaults(func=run_stopwords)


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
    parser.add_argument('--version', '-V', action='version',
                        version=f'%(prog)r version: {__slrkit_version__}')
    # dest is required to avoid a crash when the user inputs no command
    subparser = parser.add_subparsers(title='slrkit commands',
                                      required=True, dest='command')
    # init
    subparser_init(subparser)
    # import
    subparser_import(subparser)
    # journals
    subparser_journals(subparser)
    # acronyms
    subparser_acronyms(subparser)
    # preproc
    subparser_preproc(subparser)
    # terms
    subparser_terms(subparser)
    # fawoc
    subparser_fawoc(subparser)
    # postprocess
    subparser_postproc(subparser)
    # lda
    subparser_topics(subparser)
    # report
    subparser_report(subparser)
    # record
    subparser_record(subparser)
    # stopwords
    subparser_stopword(subparser)
    # build
    subparser_build(subparser)
    # readme
    subparser_readme(subparser)
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
