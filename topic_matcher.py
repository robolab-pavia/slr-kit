import json
import argparse


def init_argparser():
    parser = argparse.ArgumentParser()
    # arguments needed
    parser.add_argument("first_file", type=str, help="the first json file with the list of topics")
    # parser.add_argument("first_topic", type=str, help="the topic name from the first file")
    parser.add_argument("second_file", type=str, help="the second json file with the list of topics")
    # parser.add_argument("second_topic", type=str, help="the topic name from the second file")

    return parser


def json_opener(path1, path2):
    with open(path1) as file1:
        topics_data1 = json.load(file1)

    with open(path2) as file2:
        topics_data2 = json.load(file2)

    return topics_data1, topics_data2


def topic_matcher(topic1, topic2):
    # for every word the abs difference is computed, if word can't find a match, difference is set as 1 by default
    partial_diff = 1.0
    metric = 0.0
    counter = 0

    for term1 in topic1["terms_probability"]:
        for term2 in topic2["terms_probability"]:
            if term1 == term2:
                partial_diff = abs(topic1["terms_probability"][term1] - topic2["terms_probability"][term2])
        metric += partial_diff
        counter += 1
        partial_diff = 1.0
    # compute average difference for the topic words and transformed in a percentage notation
    metric /= counter
    percentage_metric = round((1.0 - metric) * 100.0, 2)

    return percentage_metric


def csv_writer(topics_dict1, topics_dict2):

    for topic1 in topics_dict1:
        for topic2 in topics_dict2:
            percentage_metric = topic_matcher(topics_dict1[topic1], topics_dict2[topic2])

            print(topics_dict1[topic1].get("name") + " and " + topics_dict2[topic2].get(
                "name") + " are " + str(
                percentage_metric) + " % similar")


def main():
    parser = init_argparser()
    args = parser.parse_args()

    topics_data1, topics_data2 = json_opener(args.first_file, args.second_file)
    csv_writer(topics_data1, topics_data2)


if __name__ == "__main__":
    main()
