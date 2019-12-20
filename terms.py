import csv
import enum
from dataclasses import dataclass
import utils


class Label(enum.Enum):
    """
    Label for a classified term.

    Each member contains a name and a key.
    name is a str. It should be a meaningful word describing the label.
    It goes in the csv as the term classification
    key is a str. It is the key used by the program to classify a term.

    In Label member creation the tuple is (name, default_key)
    In the definition below the NONE Label is provided to 'classify' an
    un-marked term.

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
            if label.label_name == name:
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
        self.label_name = name
        self.key = key


@dataclass
class Term:
    index: int
    term: str
    count: int
    label: Label
    order: int
    related: str

    def is_classified(self):
        """
        Tells if a Term is classified or not

        :return: True if the Term is classified, False otherwise
        :rtype: bool
        """
        return self.label != Label.NONE


class TermList(object):
    """
    :type items: list[Term] or None
    :type csv_header: list[str] or None
    """

    def __init__(self, items=None):
        """
        Creates a TermList

        :param items: a list of Term to be included in self. Default: None
        :type items: list[Term] or None
        """
        self.items = items
        self.csv_header = None

    def from_tsv(self, infile):
        """
        Gets the terms from a tsv file

        :param infile: path to the tsv file to read
        :type infile: str
        :return: the tsv header and the list of terms read by the file
        :rtype: (list[str], list[Term])
        """
        with open(infile, newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            header = csv_reader.fieldnames
            items = []
            for row in csv_reader:
                try:
                    order_value = row['order']
                    if order_value == '':
                        order = -1
                    else:
                        order = int(order_value)
                except KeyError:
                    order = -1

                related = row.get('related', '')
                try:
                    label = Label.get_from_name(row['label'])
                except KeyError:
                    print('\n{} does not contains the field "label"\n'.format(infile))
                    raise

                item = Term(
                    index=0,
                    term=row['keyword'],
                    count=row['count'],
                    label=label,
                    order=order,
                    related=related
                )
                items.append(item)

        if 'related' not in header:
            header.append('related')

        if 'order' not in csv_reader.fieldnames:
            header.append('order')

        self.csv_header = header
        self.items = items
        return header, items

    def to_tsv(self, outfile):
        """
        Saves the terms in a tsv file

        :param outfile: path to the tsv file to write the terms
        :type outfile: str
        """
        with open(outfile, mode='w') as out:
            writer = csv.DictWriter(out, fieldnames=self.csv_header,
                                    delimiter='\t', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for w in self.items:
                if w.order >= 0:
                    order = str(w.order)
                else:
                    order = ''

                item = {'keyword': w.term,
                        'count': w.count,
                        'label': w.label.label_name,
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

    def get_last_classified_term(self):
        """
        Finds the last classified term

        :return: the last classified term
        :rtype: Term or None
        """
        last = self.get_last_classified_order()
        if last < 0:
            return None
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
        :rtype: TermList
        """
        for w in self.items:
            if w.term == term:
                w.label = label
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
            if w.label != label or w.order >= 0:
                continue

            if self._str_contains(w.term, key):
                containing.append(w.term)
            else:
                not_containing.append(w.term)

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
        return len([w for w in self.items if w.label == label])

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
        for _ in utils.substring_index(string, substring):
            # if here there at least one instance of substring
            return True
        else:
            # if here the generator is empty
            return False
