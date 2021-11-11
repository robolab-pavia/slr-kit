# slr-kit: a tool to assist the writing of Systematic Literature Reviews

`slr-kit` is a tool that is intended to assist in the development of a Systematic Literature Review (SLR) of scientific papers related to a specific scientific field.

The tool is based on a specific semi-supervised workflow that leverages automatic techniques based on text mining to automatize key steps in the workflow, while some manual operations are necessary to provide the expert knowledge to the process. `slr-kit` aims at making the manual operations as efficient as possible.

It analyzes the abstracts of scientific papers to derive the topics associated to the papers, allowing the clustering of papers by topic.

As a byproduct, `slr-kit` can be used to derive a listo terms related on a specific topic.

# Table of contents

- Installation
- Usage
- The workflow on data from scratch (basic version)
- Using existing data
- Links to other docs

# Installation

## From git repository

Clone the repository:

```
git clone <repository address>
```

Install the dependencies:

```
pip install -r requirements.txt
```

# Usage

If `slr-kit` was installed from the git repository, you can launch the `slrkit.py` command:

```
python3 slrkit.py
```

# The workflow on data from scratch (basic version)

The workflow supported by `slr-kit` includes several optional steps that can be done to refine the results.
Here it is reported the mandatory steps to obtain the desired reports starting from the bibliographic material.

The basic steps are:

- initialization of the project (automatic)
- import of bibliographic data (automatic)
- preprocessing of the abstracts (automatic)
- extraction of terms (automatic)
- classification of terms using FAWOC (manual)
- topics extraction (automatic)
- generation of reports (automatic)


## Initialization of the project (automatic)

Create a new directory to host the `slr-kit` project, and move into the newly created directory (we use `my-project` as test project):

```
mkdir my-project && cd my-project
```

Initialize the project:

```
slrkit init my-project
```

## Import of bibliographic data (automatic)

Copy the bibliographic data into the project directory, naming it `my-project.ris`.

Change the `input_file` parameter into the file configuration file `slr-kit.conf/import.toml` as follows:

```
input_file = "my-project.ris"
```

At the moment, only the [RIS file format](https://en.wikipedia.org/wiki/RIS_(file_format)) is supported as source data file.
The RIS format is one of the many available on, e.g., [Scopus](https://www.scopus.com/).
Make sure that the file contains the Abstract of the papers.

To import the data into the project, run the command:

```
slrkit import
```

## Preprocessing of the abstracts (automatic)

Preprocess the abstracts:

```
slrkit preprocess
```

This performs several text processing operations on the abstracts, preparing the text for further processing.

## Extraction of terms (automatic)

Extract the list of terms from the abstracts:

```
slrkit terms
```

## Classification of terms using FAWOC (manual)

Run FAWOC and classify the terms:

```
slrkit fawoc terms
```

With FAWOC, you can quickly (well, some time is required depending on the amount of terms) classify the terms.

For this basic example, you can distinguish the terms between RELEVANT (pressing `r`) and NOISE (press `n`) terms.
If you make some mistake, press `u` to undo.

## Extraction of topics (automatic)

Run the text mining procedure with the command to generate the topics:

```
slrkit topics
```

Alternatively, you want to run an optimization in the generation of the topics with:

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

The reports are generated in the `<timestamp>_<UUID>_reports/`.
In the directory, you will find some `.md`, `.tex` and `.png` files that report the graphs and tables with the results.

# Using existing data

## Clone a repository with labelled data

```
git clone <repository>
```

## Regenerate all the necessary data

```
slrkit build
```

## Generate topics and reports

```
slrkit topics optimize
```

```
slrkit report
```

# Links to other docs

- documentation of the commands of `slr-kit`: [slrkit command documentation](slrkit.md)
- documentation of the scripts: TBD
- FAWOC: TBD
- utilities and analysis tools: TBD

