import json
import csv
import collections
from matplotlib import pyplot as plt

year = 0
paper_id = 0
topic_dic = collections.defaultdict(dict)

plt.style.use("fivethirtyeight")

with open("energy1/energy_docs-topics.json") as file:
    topics_data = json.load(file)

#print(type(topics_data[0]))

papers_data = list(csv.reader(open("energy1/energy_preproc.csv", encoding="utf8"), delimiter="\t"))

papers_dic = dict()

for paper in papers_data[1:]:
    key = int(paper[0])
    data = int(paper[2])
    papers_dic[key] = data

for entry in topics_data:
    paper_id = entry["id"]
    year = papers_dic.get(paper_id)
    for topic in entry["topics"]:
        topic_dic[topic][year] = topic_dic[topic].get(year, 0) + 1

for dic in topic_dic:

    sorted_dic = sorted(topic_dic[dic].items())
    x, y = zip(*sorted_dic)

    plt.grid(True)
    plt.plot(x, y)
    plt.title("topic number: "+dic)
    plt.xlabel("Year")
    plt.ylabel("# of papers")
    plt.tight_layout()
    plt.show()






