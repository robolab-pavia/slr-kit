import json
import argparse

def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    epilog = """"'The program uses two files: <dataset>/<prefix>_preproc.csv and
 '<dataset>/<prefix>_terms.csv.
 It outputs the topics in <dataset>/<prefix>_topics.json and the topics assigned
 to each document in <dataset>/<prefix>_docs-topics.json"""
    parser = argparse.ArgumentParser(description='Performs the LDA on a dataset',
                                     epilog=epilog)
    parser.add_argument('topics', action="store", type=str,
                        help='JSON file containing the topics.')
    parser.add_argument('docs_topics', action="store", type=str,
                        help='JSON file containing the association between docs and topics.')
    return parser




def main():
    args = init_argparser().parse_args()

    with open(args.topics) as topics_file:
        loaded_topics = json.load(topics_file)


    # for each topics keeps the title and the three highest terms
    topics = {}
    for p in loaded_topics:
        identifier = p
        loaded_prob = loaded_topics[p]["terms_probability"]
        probab = []
        for pp in loaded_prob:
            probab.append((pp, loaded_prob[pp]))
        sorted_by_second = sorted(probab, key=lambda tup: tup[1], reverse=True)
        topics[identifier] = {
                "name": loaded_topics[p]["name"],
                "terms": sorted_by_second[:5]
        }

    #print(topics)

    with open(args.docs_topics) as docs_file:
        docs = json.load(docs_file)

    #print(docs)

    for d in docs:
        print(d['title'])
        top = []
        for t in d['topics']:
            top.append((t, d['topics'][t]))
        sorted_by_second = sorted(top, key=lambda tup: tup[1], reverse=True)
        for t in sorted_by_second:
            key = t[0]
            name = topics[key]['name']
            terms = [t[0] for t in topics[key]['terms']]
            print("  {:5.1f}% {} {}".format(t[1] * 100, name, terms))


if __name__ == '__main__':
    main()
