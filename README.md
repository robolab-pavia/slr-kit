# slr-kit: a tool to assist the writing of Systematic Literature Reviews

`slr-kit` is a tool that is intended to assist in the development of a Systematic Literature Review (SLR) of scientific papers related to a specific scientific field.

The tool is based on a semi-supervised workflow that leverages automatic techniques based on text mining to automatize key steps in the workflow, while some manual operations are necessary to provide the expert knowledge to the process. `slr-kit` aims at making the manual operations as efficient as possible.

It analyzes the abstracts of scientific papers to derive the topics associated to the papers, allowing the clustering of papers by topic and the derivation of useful statistics about the set of papers and their topics, such as the trend of publications per topic during the years, and the amount of papers published on different journals by topic.

As a byproduct, `slr-kit` can be used to manage a list of terms related on a specific topic, curated by an expert, which can be useful in many NLP applications.

# Table of contents

- Installation
- Usage
- The workflow on data from scratch (basic version)
- Using existing data
- Links to other docs

# Installation

## Using pip

`slrkit` is available on PyPi and can thus be installed as

```
pip install slrkit
```

The documentation below refers to the installed program though pip.

## From git repository

Clone the repository:

```
git clone <repository address>
```

Install the dependencies:

```
pip install -r requirements.txt
```

In this case, the command can be run, from within the repository directory, as

```
python3 slrkit/slrkit.py
```

# Install `nltk` corpora

After installing `slrkit` and its dependencies, the corpora for `NLTK` must be downloaded.
The following command can be used:

```
python -m nltk.downloader popular
```

# Usage

If `slr-kit` was installed from the git repository, you can launch the `slrkit` command.
Use the `-h` or `--help` option to generate the following help:

```
$ slrkit --help
usage: slrkit [-h] [-C path] [--version]
              {init,import,journals,acronyms,preprocess,terms,fawoc,postprocess,topics,report,record,stopwords,build,readme}
                 ...

slrkit project handling tool

options:
  -h, --help            show this help message and exit
  -C path               Change directory to 'path' before running the specified command.
  --version, -V         show program's version number and exit

slrkit commands:
  {init,import,journals,acronyms,preprocess,terms,fawoc,postprocess,topics,report,record,stopwords,build,readme}
    init                Initialize a slr-kit project
    import              Import a bibliographic database converting to the csv format used by slr-kit.
    journals            Subcommand to extract and filter a list of journals. This command accepts a
                        subcommand. If none is given, "extract" is assumed.
    acronyms            Extract acronyms from texts.
    preprocess          Run the preprocess stage in a slr-kit project
    terms               Subcommand to extract and handle lists of terms in a slr-kit project. This
                        command accepts a subcommand. If none is given, "generate" is assumed.
    fawoc               Run fawoc in a slr-kit project. This command accepts a subcommand. If none is
                        given "terms" is assumed.
    postprocess         Run the postprocess stage in a slr-kit project
    topics              Subcommand to extract the topics from the documents in a slrkit project. This
                        command requires a subcommand.
    report              Run the report creation script in a slr-kit project.
    record              Record a snapshot of the project in the underlying git repository
    stopwords           Extracts the terms classified as stopwords from the terms file
    build               Run all the commands needed to re-create all the files after cloning a slr-kit
                        project.
    readme              Creates a README.md file for the project using information from the META.toml
                        file. The README.md is also committed to the git repository.
```

Each sub-command is explained in details the [slrkit.md](docs/scripts.md) documentation file.

# The workflow on data from scratch (basic version)

The organization of the information is based on _projects_.
A project is basically a directory containing all the necessary data related to the SLR.
Such data are manipulated using the `slrkit` commands.

All the data, being final or intermediate results, are stored in easy to inspect CSV files.

The workflow supported by `slr-kit` includes several optional steps that can be done to refine the final results.
Here it is reported the mandatory steps to obtain the desired reports starting from the bibliographic material.

The basic steps are:

- initialization of the project (automatic)
- import of bibliographic data (automatic)
- preprocessing of the abstracts (automatic)
- extraction of terms (automatic)
- classification of terms using FAWOC (manual)
- topics extraction (automatic)
- generation of reports (automatic)

Anytime during the various steps, you can run the command

```
slrkit record
```

that will produce a snapshot of the current information required by an slr-kit project.
Note that the snapshots are maintained as standard git repositories.
Therefore, you can use the regular git commands to operate on the repository (git log, git status, git checkout, etc.).
Moreover, you can use the usual git push/pull to move the dataset to/from remotes.

## Initialization of the project (automatic)

Create a new directory to host the `slr-kit` project, and move into the newly created directory (we use `my-project` as example project):

```
mkdir my-project && cd my-project
```

Initialize the project:

```
slrkit init my-project
```

## Import of bibliographic data (automatic)

At the moment, only the [RIS file format](https://en.wikipedia.org/wiki/RIS_(file_format)) is supported as source data file.
The RIS format is one of the many available on, e.g., [Scopus](https://www.scopus.com/).
Make sure that the file contains the Abstract of the papers.

Copy the bibliographic data into the project directory, naming it `my-project.ris`.

To make `slrkit` aware of the name of the bibliographic data, set the value of the `input_file` parameter into the file configuration file `slr-kit.conf/import.toml` (this file was generated by the `init` command) as follows:

```
input_file = "my-project.ris"
```

To import the data into the project, run the command:

```
slrkit import
```

## Preprocessing of the abstracts (automatic)

This operation performs several text processing operations on the abstracts, such as stemming and removal of punctuation, preparing the text for further processing.

The command to run the preprocessing is:

```
slrkit preprocess
```

## Extraction of terms (automatic)

Extract the list of terms from the abstracts:

```
slrkit terms
```

## Classification of terms using FAWOC (manual)

Run FAWOC and manually classify the terms:

```
slrkit fawoc terms
```

FAWOC is an interactive terminal program that allows to quickly (well, some time is required depending on the amount of terms) classify the terms.

For this basic example, you can distinguish the terms between RELEVANT (pressing `r`) and NOISE (press `n`) terms.
If you make some mistake, press `u` to undo.

## Postprocessing (automatic)

This operation processes the documents by managing the terms classified in the previous stage.
Run it with

```
slrkit postprocess
```

## Extraction of topics (automatic)

Run the text mining procedure with the command to generate the topics:

```
slrkit topics
```

Alternatively, you may want to run an optimization in the generation of the topics with:

```
slrkit topics optimize
```

This operation takes **significantly more time**, but produces results that are **far more accurate**.

HINT: Use the optimization everytime a "production ready" result is required.

## Generation of reports (automatic)

Generate the reports with:

```
slrkit report
```

The reports are stored in a newly generated unique directory whose name has the format `<timestamp>_<UUID>_reports/`.
In the directory, you will find some `.md`, `.tex` and `.png` files that report the graphs and tables with the results.
The reports are generated according to some templates in Markdown and LaTeX format for being easily edited or directly included in scientific papers or websites.

# Using existing data

slr-kit can also be used on an existing dataset, i.e., a set of papers that were collected by someone else and already (even partially) elaborated.

In this example, we assume that the dataset was properly recorded using `slrkit record`, and all the terms have been already classified.

## Clone a repository with labelled data

A dataset is kept versioned using git.
Therefore, if the dataset is shared in some online repository, you can simply download it with

```
git clone <repository>
```

A repository containing the classification of a dataset about Natural Language Processing is available here: [https://github.com/robolab-pavia/slrkit_NLP_Nocera](https://github.com/robolab-pavia/slrkit_NLP_Nocera).

## Regenerate all the necessary data

First, you need to re-build all the preprocessed data.
This is done with one single command:

```
slrkit build
```

At this point, if necessary, you can refine the classification of terms by running the usual command

```
slrkit fawoc terms
```

## Generate topics and reports

Now you are ready to generate the topics and the reports with the same commands explained above:

```
slrkit topics optimize
```

followed by

```
slrkit report
```

# Links to other docs

- documentation of the [commands of `slr-kit`](docs/slrkit.md)
- documentation of the [scripts](docs/scripts.md)
- [FAWOC](https://github.com/robolab-pavia/fawoc)
- utilities and analysis tools: TBA

