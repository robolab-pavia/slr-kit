import argparse
import collections
import datetime
import json
import math
import os
import pathlib
import shutil
import warnings
from itertools import islice

from RISparser import readris
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from tabulate import tabulate

from slrkit_utils.argument_parser import ArgParse

YEARTOPIC_TEX = 'yeartopic.tex'
JOURNALTOPIC_TEX = 'journaltopic.tex'
JOURNALYEAR_TEX = 'journalyear.tex'
MD_TEMPLATE = 'report_template.md'
MD_REPORT = 'report.md'
TEX_TEMPLATE = 'report_template.tex'
TEX_REPORT = 'report.tex'
YEARFIGURE = 'reportyear.png'
TEMPLATES_DIRNAME = 'report_templates'
TABLES_DIRNAME = 'tables'
PLACEHOLDERFIGURE = 'placeholder.png'


def get_journal(paper):
    """
    Returns the name of the journal from a dict loaded from RIS.
    Raises an exception if no suitable keys are found.

    :return: The name of the journal.
    :rtype: str
    """
    if 'secondary_title' in paper:
        return paper['secondary_title']
    if 'custom3' in paper:
        return paper['custom3']
    raise KeyError(paper)


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = ArgParse()
    parser.add_argument('ris_file', type=str, suggest_suffix='.ris',
                        help='the path to the ris file containing papers data',
                        input=True)
    parser.add_argument('json_file', type=str, cli_only=True,
                        help='the path to the lda results file containing the '
                             'association between documents and topics.')
    parser.add_argument('--dir', '-d', metavar='FILENAME',
                        help='output directory where reports and files will be '
                             'saved')
    parser.add_argument('--minyear', '-m', type=int,
                        help='minimum year to be reported. '
                             'If missing, the minimum year '
                             'in the data is used.')
    parser.add_argument('--maxyear', '-M', type=int,
                        help='maximum year to be reported. '
                             'If missing, the maximum year '
                             'in the data is used.')
    parser.add_argument('--plotsize', '-p', type=int,
                        help='number of topics to be displayed in each subplot.')

    return parser


def prepare_papers(ris_path, json_path):
    """
    Extracts all data for each paper in the ris file and the json file to create
    a list of dictionaries containing all grouped data.

    :param ris_path: path to the ris file containing papers data
    :type ris_path: str
    :param json_path: path to json file containing lda results
    :type json_path: str
    :return: the list of dictionaries and the list of topics
    :rtype: tuple[list, list]
    """
    with open(json_path) as file:
        papers_with_topics = json.load(file)

    papers_from_ris = []
    with open(ris_path, 'r', encoding='utf-8') as bibliography_file:
        entries = readris(bibliography_file)
        for entry in entries:
            if 'title' not in entry:
                continue
            papers_from_ris.append(entry)

    good_papers = []
    for paper in papers_from_ris:
        for paper_data in papers_with_topics:
            if paper['title'] == paper_data['title']:
                topics = paper_data['topics']
                paper['topics'] = topics
                good_papers.append(paper)
                break

    topics_list = []
    for paper in good_papers:
        for key in paper['topics']:
            if int(key) not in topics_list:
                topics_list.append(int(key))

    topics_list.sort()

    return good_papers, topics_list


def report_year(papers_list, topics_list):
    """
    Creates a dictionary with number of papers published each year for every topic.
    Papers are weighted by coherence.

    :param papers_list: list of dictionaries with data for every paper
    :type papers_list: list
    :param topics_list: list of topics
    :type topics_list: list
    :return: dictionary with topic-year data
    :rtype: dict
    """

    topics_dict = collections.defaultdict(dict)

    for topic_id in topics_list:
        for paper in papers_list:
            topics = paper['topics']
            if str(topic_id) in topics:
                year = int(paper['year'])
                topics_dict[topic_id][year] = (topics_dict[topic_id].get(year, 0)
                                               + float(topics[str(topic_id)]))

    return topics_dict


def prepare_journals(papers_list):
    """
    Check how many papers were published by each journal.

    :param papers_list: list with data of every paper
    :type papers_list: list
    :return: list of journals and their number of publications
    :rtype: list
    """

    journals = []
    journals_dict = {}
    for paper in papers_list:
        journal_name = get_journal(paper)
        if journal_name not in journals:
            journals.append(journal_name)

    for journal in journals:
        for paper in papers_list:
            journal_name = get_journal(paper)
            if journal_name == journal:
                journals_dict[journal] = journals_dict.get(journal, 0) + 1

    journals_dict = sorted(journals_dict.items(), key=lambda x: x[1],
                           reverse=True)

    return journals_dict


def report_journal_topics(journals_dict, papers_list):
    """
    Creates a dictionary with for each journal the number of papers published for
    each topic.

    :param journals_dict: list of dictionary with data of journals and their publications
    :type journals_dict: list
    :param papers_list: list with data of every paper
    :type papers_list: list
    :return: dictionary with journal-topic data
    :rtype: dict
    """

    journal_topic = collections.defaultdict(dict)
    for journal, _ in journals_dict[0:10]:
        for paper in papers_list:
            journal_title = get_journal(paper)
            if journal_title == journal:
                topics = paper['topics']
                for topic in topics:
                    journal_topic[journal][topic] = (journal_topic[journal].get(topic, 0)
                                                    + topics[topic])

    return journal_topic


def report_journal_years(papers_list, journals_dict):
    """
    Creates a dictionary with for each journal the number of papers published for
    each year.

    :param papers_list: list with data of every paper
    :type papers_list: list
    :param journals_dict: list of dictionary with data of journals and their publications
    :type journals_dict: list
    :return: dictionary with journal-year data and the min and max year in the dict
    :rtype: tuple(dict, int, int)
    """
    journal_year = collections.defaultdict(dict)
    min_year = float('inf')
    max_year = -1
    for journal, value in journals_dict[0:10]:
        for paper in papers_list:
            journal_title = get_journal(paper)
            if journal_title == journal:
                year = int(paper['year'])
                journal_year[journal][year] = journal_year[journal].get(year, 0) + 1
                if year > max_year:
                    max_year = year
                if year < min_year:
                    min_year = year

    return journal_year, min_year, max_year


def plot_years(topics_dict, dirname, plot_size, templates):
    """
    Creates a plot for the number of papers published each year for each topic

    :param topics_dict: dictionary with topic-year data
    :type topics_dict: dict
    :param dirname: name of the directory where graph will be saved
    :type dirname: Path
    :param plot_size: number of topics per plot
    :type plot_size: int
    :param templates: name of directory where templates are saved
    :type templates: Pat
    """
    shutil.copy(templates / PLACEHOLDERFIGURE, dirname / YEARFIGURE)

    for k in topics_dict:
        if len(topics_dict[k]) == 0:
            warnings.warn('There was a problem with the dictionary of topics.')
            return

    rows = math.ceil(len(topics_dict) / plot_size)
    for i in range(rows):
        for topic in islice(topics_dict, i*plot_size, (i+1)*plot_size):
            sorted_dic = sorted(topics_dict[topic].items())
            x, y = zip(*sorted_dic)

            plt.plot(x, y, label='topic ' + str(topic))
            plt.title('topics yearly graph (part {0})'.format(i + 1))
            plt.xlabel('Year')
            plt.ylabel('# of papers (weighted by coherence)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(dirname / 'reportyear{0}'.format(i+1))

        plt.clf()

    fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(8, 4*rows))

    for i in range(rows):
        for topic in islice(topics_dict, i * plot_size, (i + 1) * plot_size):
            sorted_dic = sorted(topics_dict[topic].items())
            x, y = zip(*sorted_dic)
            ax[i].plot(x, y, label='topic ' + str(topic))
            ax[i].grid(True)
            ax[i].set_title('topics yearly graph (part {0})'.format(i+1))
            ax[i].set_xlabel('Year')
            ax[i].set_ylabel('# of papers (weighted by coherence)')
            ax[i].legend()

    fig.tight_layout()
    plt.savefig(dirname / YEARFIGURE)
    plt.clf()


def create_topic_year_list(topics_dict, max_year, min_year):
    first_line = ['Topic']
    first_line.extend(list(range(min_year, max_year + 1)))
    topic_year_list = [first_line]
    for topic in topics_dict:
        sorted_dic = sorted(topics_dict[topic].items())
        line = [topic]
        x, y = zip(*sorted_dic)
        for year in first_line[1:]:
            if year in x:
                line.append('{:.2f}'.format(y[x.index(year)]))
            else:
                line.append(0)
        topic_year_list.append(line)
    return topic_year_list


def create_journal_topic_list(journals_topic, topics_dict):
    first_line = ['Journal']
    topics_list = list(range(0, len(topics_dict)))
    first_line.extend(topics_list)
    journal_topic_list = [first_line]
    for journal in journals_topic:
        line = [journal]
        sorted_dic = sorted(journals_topic[journal].items())
        x, y = zip(*sorted_dic)
        for topic in first_line[1:]:
            if str(topic) in x:
                line.append('{:.2f}'.format(y[x.index(str(topic))]))
            else:
                line.append(0)
        journal_topic_list.append(line)
    return journal_topic_list


def create_journal_year_list(journals_year, max_year, min_year):
    first_line = ['Journal']
    first_line.extend(list(range(min_year, max_year + 1)))
    journal_year_list = [first_line]
    for journal in journals_year:
        sorted_dic = sorted(journals_year[journal].items())
        line = [journal]
        x, y = zip(*sorted_dic)
        for year in first_line[1:]:
            if year in x:
                line.append(y[x.index(year)])
            else:
                line.append(0)
        journal_year_list.append(line)
    return journal_year_list


def save_markdown_report(topic_year_list, journal_topic_list, journal_year_list,
                         md_filename, template_path):
    env = Environment(loader=FileSystemLoader(template_path), autoescape=True)
    template = env.get_template(MD_TEMPLATE)
    topic_year_table = tabulate(topic_year_list, headers='firstrow',
                                tablefmt='github')
    journal_topic_table = tabulate(journal_topic_list, headers='firstrow',
                                   floatfmt='.3f', tablefmt='github')
    journal_year_table = tabulate(journal_year_list, headers='firstrow',
                                  floatfmt='.3f', tablefmt='github')
    md_file = template.render(year_report=YEARFIGURE,
                              year_table=topic_year_table,
                              journal_topic_table=journal_topic_table,
                              journal_year_table=journal_year_table)

    with open(md_filename, 'w') as fh:
        fh.write(md_file)


def save_latex_table(data_list, filename, fix_align):
    latex_table = tabulate(data_list, headers='firstrow', tablefmt='latex')
    if fix_align:
        latex_table = latex_table.replace('lrrrrr', 'p{3cm}rrrrr')
    with open(filename, 'w') as f:
        f.write(latex_table)


def prepare_tables(topics_dict, journals_topic, journals_year, dirname,
                   md_template_path, min_year, max_year):
    """
    Creates tables for every data created in previous function, with the Tabulate
    module and saves them

    :param topics_dict: dictionary with topic-year data
    :type topics_dict: dict
    :param journals_topic: dictionary with journal-topic data
    :type journals_topic: dict
    :param journals_year: dictionary with journal-year data
    :type journals_year: dict
    :param dirname: name of the directory where the files will be saved
    :type dirname: pathlib.Path
    :param md_template_path: path to the directory containing the markdown
        template
    :type md_template_path: pathlib.Path
    :param min_year: minimum year that will be used in the report
    :type min_year: int
    :param max_year: maximum year that will be used in the report
    :type max_year: int
    """
    tables: pathlib.Path = dirname / TABLES_DIRNAME
    tables.mkdir(exist_ok=True)

    topic_year_list = create_topic_year_list(topics_dict, max_year, min_year)
    save_latex_table(topic_year_list, tables / YEARTOPIC_TEX, False)

    journal_topic_list = create_journal_topic_list(journals_topic, topics_dict)
    save_latex_table(journal_topic_list, tables / JOURNALTOPIC_TEX, True)

    journal_year_list = create_journal_year_list(journals_year, max_year,
                                                 min_year)
    save_latex_table(journal_year_list, tables / JOURNALYEAR_TEX, True)

    save_markdown_report(topic_year_list, journal_topic_list, journal_year_list,
                         dirname / MD_REPORT, md_template_path)


def report(args):
    script_dir = pathlib.Path(__file__).parent
    cwd = pathlib.Path.cwd()
    listdir = os.listdir(cwd)
    templates = script_dir / TEMPLATES_DIRNAME
    if MD_TEMPLATE not in listdir:
        shutil.copy(templates / MD_TEMPLATE, cwd)
    if TEX_TEMPLATE not in listdir:
        shutil.copy(templates / TEX_TEMPLATE, cwd)

    ris_path = args.ris_file
    json_path = args.json_file

    if args.dir is not None:
        dirname = cwd / args.dir
    else:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dirname = cwd / ('report' + timestamp)

    dirname.mkdir(exist_ok=True)

    papers_list, topics_list = prepare_papers(ris_path, json_path)
    topics_dict = report_year(papers_list, topics_list)

    if args.plotsize is not None:
        plot_size = args.plotsize
    else:
        plot_size = 10

    plot_years(topics_dict, dirname, plot_size, templates)
    journals_dict = prepare_journals(papers_list)
    journals_year, min_year, max_year = report_journal_years(papers_list,
                                                             journals_dict)
    journals_topics = report_journal_topics(journals_dict, papers_list)
    if args.minyear is not None:
        min_year = args.minyear
    if args.maxyear is not None:
        max_year = args.maxyear

    if min_year > max_year:
        msg = 'The minimum year {} is greater than the maximum year {}'
        raise ValueError(msg.format(min_year, max_year))

    prepare_tables(topics_dict, journals_topics, journals_year, dirname, cwd,
                   min_year, max_year)

    shutil.copy(cwd / TEX_TEMPLATE, dirname / TEX_REPORT)


def main():
    parser = init_argparser()
    args = parser.parse_args()
    report(args)


if __name__ == '__main__':
    main()
