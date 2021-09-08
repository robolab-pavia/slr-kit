import json
import argparse


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Joins the output of LDA in a '
                                                 'single file')
    parser.add_argument('terms_topics', action="store", type=str,
                        help='JSON file containing the association between '
                             'terms and topics.')
    parser.add_argument('docs_topics', action="store", type=str,
                        help='JSON file containing the association between docs'
                             ' and topics.')
    parser.add_argument('--output', '-o', action="store", type=str,
                        help='Output file')
    return parser


def join_lda_info(loaded_topics, docs, output):
    """
    Joins the information about the topics assigned to each document and outputs them

    :param loaded_topics: information about the topics
    :type loaded_topics: dict
    :param docs: association between documents and topics
    :type docs: list[dict]
    :param output: output file. If None, stdout is used
    :type output: str or None
    """
    # for each topics keeps the title and the three highest terms
    topics = {}
    for p in loaded_topics:
        identifier = p
        loaded_prob = loaded_topics[p]['terms_probability']
        probab = []
        for pp in loaded_prob:
            probab.append((pp, loaded_prob[pp]))
        sorted_by_second = sorted(probab, key=lambda tup: tup[1], reverse=True)
        topics[identifier] = {
            'name': loaded_topics[p]['name'],
            'terms': sorted_by_second[:5]
        }

    if output is not None:
        output_file = open(output, 'w')
    else:
        output_file = None

    for d in docs:
        print(d['title'], file=output_file)
        top = []
        for t in d['topics']:
            top.append((t, d['topics'][t]))
        sorted_by_second = sorted(top, key=lambda tup: tup[1], reverse=True)
        for t in sorted_by_second:
            key = t[0]
            name = topics[key]['name']
            terms = [t[0] for t in topics[key]['terms']]
            print('  {:5.1f}% {} {}'.format(t[1] * 100, name, terms),
                  file=output_file)


def main():
    args = init_argparser().parse_args()
    terms_topics = args.terms_topics
    docs_topics = args.docs_topics
    output = args.output

    with open(terms_topics) as topics_file:
        loaded_topics = json.load(topics_file)

    with open(docs_topics) as docs_file:
        docs = json.load(docs_file)

    join_lda_info(loaded_topics, docs, output)


if __name__ == '__main__':
    main()
