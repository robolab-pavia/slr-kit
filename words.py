import csv
import enum
from dataclasses import dataclass


class WordClass(enum.Enum):
    """
    Class for a classified word.

    Each member contains a classname and a key.
    classname is a str. It should be a meaningful word describing the class.
    It goes in the csv as the word classification
    key is a str. It is the key used by the program to classify a word.

    In WordClass memeber creation the tuple is (classname, default_key)
    In the definition below the NONE WordClass is provided to 'classify' an
    un-marked word.
    """
    NONE = ('', '')
    KEYWORD = ('keyword', 'k')
    NOISE = ('noise', 'n')
    RELEVANT = ('relevant', 'r')
    NOT_RELEVANT = ('not-relevant', 'x')
    POSTPONED = ('postponed', 'p')

    @staticmethod
    def get_from_key(key: str):
        for clname in WordClass:
            if clname.key == key:
                return clname

        raise ValueError('"{}" is not a valid key'.format(key))

    @staticmethod
    def get_from_classname(classname):
        for clname in WordClass:
            if clname.classname == classname:
                return clname

        raise ValueError('"{}" is not a valid class name'.format(classname))

    def __init__(self, classname, key):
        self.classname = classname
        self.key = key


@dataclass
class Word:
    index: int
    word: str
    count: int
    group: WordClass
    order: int
    related: str

    def is_grouped(self):
        return self.group != WordClass.NONE


class WordList(object):
    def __init__(self, items=None):
        self.items = items
        self.csv_header = None

    def from_csv(self, infile):
        with open(infile, newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            header = csv_reader.fieldnames
            items = []
            for row in csv_reader:
                order_value = row['order']
                if order_value == '':
                    order = None
                else:
                    order = int(order_value)

                related = row.get('related', '')
                try:
                    group = WordClass.get_from_classname(row['group'])
                except ValueError:
                    group = WordClass.get_from_key(row['group'])

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
        with open(outfile, mode='w') as out:
            writer = csv.DictWriter(out, fieldnames=self.csv_header,
                                    delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for w in self.items:
                item = {'keyword': w.word,
                        'count': w.count,
                        'group': w.group.classname,
                        'order': w.order,
                        'related': w.related}
                writer.writerow(item)

    def get_last_inserted_order(self):
        orders = [w.order for w in self.items if w.order is not None]
        if len(orders) == 0:
            order = -1
        else:
            order = max(orders)

        return order

    def get_last_inserted_word(self):
        last = self.get_last_inserted_order()
        for w in self.items:
            if w.order == last:
                return w
        else:
            return None

    def mark_word(self, word, marker, order, related=''):
        for w in self.items:
            if w.word == word:
                w.group = marker
                w.order = order
                w.related = related
                break

        return self

    def return_related_items(self, key):
        containing = []
        not_containing = []
        for w in self.items:
            if w.is_grouped():
                continue

            if self._word_contains(w.word, key):
                containing.append(w.word)
            else:
                not_containing.append(w.word)

        return containing, not_containing

    def count_classified(self):
        return len([item for item in self.items if item.is_grouped()])

    def count_by_class(self, cls):
        return len([w for w in self.items if w.group == cls])

    @staticmethod
    def _word_contains(string, substring):
        return any([substring == word for word in string.split()])
