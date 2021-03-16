import json
import argparse


def init_argparser():
    parser = argparse.ArgumentParser()
    # arguments needed
    parser.add_argument("first_file", type=str, help="the first json file with the list of topics")
    parser.add_argument("first_topic", type=str, help="the topic name from the first file")
    parser.add_argument("second_file", type=str, help="the second json file with the list of topics")
    parser.add_argument("second_topic", type=str, help="the topic name from the second file")

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


def main():
    parser = init_argparser()
    args = parser.parse_args()

    topics_data1, topics_data2 = json_opener(args.first_file, args.second_file)
    percentage_metric = topic_matcher(topics_data1[args.first_topic], topics_data2[args.second_topic])

    print(topics_data1[args.first_topic].get("name") + " and " + topics_data2[args.second_topic].get("name") + " are " + str(
        percentage_metric) + " % similar")


if __name__ == "__main__":
    main()