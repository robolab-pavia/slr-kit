import csv
import enum
from dataclasses import dataclass


class Label(enum.Enum):
    """
    Label for a classified word.

    Each member contains a name and a key.
    name is a str. It should be a meaningful word describing the label.
    It goes in the csv as the word classification
    key is a str. It is the key used by the program to classify a word.

    In Label member creation the tuple is (name, default_key)
    In the definition below the NONE Label is provided to 'classify' an
    un-marked word.

    :type name: str
    :type key: str
    """
    NONE = ('', '')
    KEYWORD = ('keyword', 'k')
    NOISE = ('noise', 'n')
    RELEVANT = ('relevant', 'r')
    NOT_RELEVANT = ('not-relevant', 'x')
    POSTPONED = ('postponed', 'p')

    @staticmethod
    def get_from_key(key):
        """
        Searches the Label associated with a specified key

        :param key: the associated to the Label
        :type key: str
        :return: the Label associated with key
        :rtype: Label
        """
        for label in Label:
            if label.key == key:
                return label

        raise ValueError('"{}" is not a valid key'.format(key))

    @staticmethod
    def get_from_name(name):
        """
        Searches the Label associated with a specified name

        :param name: the associated to the Label
        :type name: str
        :return: the Label associated with name
        :rtype: Label
        """
        for label in Label:
            if label.name == name:
                return label

        raise ValueError('"{}" is not a valid label name'.format(name))

    def __init__(self, name, key):
        """
        Creates Label and sets its name and key

        It is meant to be used by the internals of Enum. Using it directly will
        probably result in an Exception
        :param name: name of the label
        :type name: str
        :param key: key associated to the label
        :type key: str
        """
        self.name = name
        self.key = key


@dataclass
class Word:
    index: int
    word: str
    count: int
    group: Label
    order: int
    related: str

    def is_classified(self):
        """
        Tells if a Term is classified or not

        :return: True if the Term is classified, False otherwise
        :rtype: bool
        """
        return self.group != Label.NONE


class WordList(object):
    """
    :type items: list[Word] or None
    :type csv_header: list[str] or None
    """

    def __init__(self, items=None):
        """
        Creates a TermList

        :param items: a list of Term to be included in self. Default: None
        :type items: list[Word] or None
        """
        self.items = items
        self.csv_header = None

    def from_csv(self, infile):
        """
        Gets the terms from a csv file

        :param infile: path to the csv file to read
        :type infile: str
        :return: the csv header and the list of terms read by the file
        :rtype: (list[str], list[Word])
        """
        with open(infile, newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            header = csv_reader.fieldnames
            items = []
            for row in csv_reader:
                order_value = row['order']
                if order_value == '':
                    order = -1
                else:
                    order = int(order_value)

                related = row.get('related', '')
                try:
                    group = Label.get_from_name(row['group'])
                except ValueError:
                    group = Label.get_from_key(row['group'])

                item = Word(
                    index=0,
                    word=row['keyword'],
                    count=row['count'],
                    group=group,
                    order=order,
                    related=related
                )
                items.append(item)

        if 'related' not in header:
            header.append('related')

        self.csv_header = header
        self.items = items
        return header, items

    def to_csv(self, outfile):
        """
        Saves the terms in a csv file

        :param outfile: path to the csv file to write the terms
        :type outfile: str
        """
        with open(outfile, mode='w') as out:
            writer = csv.DictWriter(out, fieldnames=self.csv_header,
                                    delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for w in self.items:
                if w.order >= 0:
                    order = str(w.order)
                else:
                    order = ''

                item = {'keyword': w.word,
                        'count': w.count,
                        'group': w.group.name,
                        'order': order,
                        'related': w.related}
                writer.writerow(item)

    def get_last_classified_order(self):
        """
        Finds the classification order of the last classified term

        :return: the classification order of the last classified term
        :rtype: int
        """
        order = max(w.order for w in self.items)
        if order < 0:
            order = -1

        return order

    def get_last_classified_word(self):
        """
        Finds the last classified term

        :return: the last classified term
        :rtype: Word
        """
        last = self.get_last_classified_order()
        for w in self.items:
            if w.order == last:
                return w
        else:
            return None

    def classify_term(self, term, label, order, related=''):
        """
        Classifies a term

        This method adds label, classification order and related term to the
        Term in self.items that represents the string term.
        term is the string representation of the Term that will be classified.
        The method searches a Term t in self.items such that t.term == term.
        order must be an int. An int less than 0 indicates no classification
        order.
        related is a string indicating the active related term at the moment of
        classification. It defaults to the empty string.
        This method return self.

        :param term: the term to classify
        :type term: str
        :param label: the Label to be assigned to the term
        :type label: Label
        :param order: the classification order
        :type order: int
        :param related: related term (if any). Default: ''
        :type related: str
        :return: self
        :rtype: WordList
        """
        for w in self.items:
            if w.word == term:
                w.group = label
                w.order = order
                w.related = related
                break

        return self

    def return_related_items(self, key, label=Label.NONE):
        """
        Searches related items in self and returns the resulting partition

        This method splits self.items in two list: the first one with all the
        strings that contains the substring key; the second one with all the
        strings that not contain key.
        Only the terms with the specified label are considered.
        The method returns two lists of strings.
        :param key: the substring to find in the terms in self.items
        :type key: str
        :param label: label to consider
        :type label: Label
        :return: the partition of the items in self based on key
        :rtype: (list[str], list[str])
        """
        containing = []
        not_containing = []
        for w in self.items:
            if w.group != label or w.order is not None:
                continue

            if self._str_contains(w.word, key):
                containing.append(w.word)
            else:
                not_containing.append(w.word)

        return containing, not_containing

    def count_classified(self):
        """
        Counts the classified terms

        :return: the number of classified terms
        :rtype: int
        """
        return len([item for item in self.items if item.is_classified()])

    def count_by_label(self, label):
        """
        Counts the terms classified as label

        :param label: the label to search
        :type label: Label
        :return: the number of terms classified as label
        :rtype: int
        """
        return len([w for w in self.items if w.group == label])

    @staticmethod
    def _str_contains(string, substring):
        """
        Tells if string contains substring

        :param string: the string to analize
        :type string: str
        :param substring: the substring to search
        :type substring: str
        :return: True if string contains substring
        :rtype: bool
        """
        return any([substring == word for word in string.split()])
