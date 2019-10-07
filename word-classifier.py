import argparse
import csv
import curses
import enum
import logging
from dataclasses import dataclass


# List of class names
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
        for clname in ClassNames:
            if clname.key == key:
                return clname

        raise ValueError('"{}" is not a valid key'.format(key))

    @staticmethod
    def get_from_classname(classname):
        for clname in ClassNames:
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

            if find_word(w.word, key):
                containing.append(w.word)
            else:
                not_containing.append(w.word)

        return containing, not_containing

    def count_classified(self):
        return len([item for item in self.items if item.is_grouped()])

    def count_by_class(self, cls):
        return len([w for w in self.items if w.group == cls])


class Win(object):
    """Contains the list of lines to display."""

    def __init__(self, group, title='', rows=3, cols=30, y=0, x=0,
                 show_title=False):
        self.group = group
        self.title = title
        self.rows = rows
        self.cols = cols
        self.x = x
        if show_title:
            self.y = y + 1
            self.win_title = curses.newwin(1, self.cols, y, self.x)
            self.win_title.addstr(self.title)
        else:
            self.y = y
            self.win_title = None

        self.win_handler = curses.newwin(self.rows, self.cols, self.y, self.x)
        self.win_handler.border()
        self.win_handler.refresh()
        self.lines = []

    def display_lines(self, rev=True, highlight_word='', color_pair=1):
        if rev:
            word_list = reversed(self.lines)
        else:
            word_list = self.lines
        i = 0
        for w in word_list:
            trunc_w = w[:self.cols - 2]
            if highlight_word == '':
                self.win_handler.addstr(i + 1, 1, trunc_w + ' ' * (self.cols - 2 - len(trunc_w)))
            else:
                tok = w.split(highlight_word)
                self.win_handler.addstr(i + 1, 1, '')
                for t in tok:
                    self.win_handler.addstr(t)
                    self.win_handler.addstr(highlight_word,
                                            curses.color_pair(color_pair))
                self.win_handler.addstr(i + 1, len(trunc_w) + 1, ' ' * (self.cols - 2 - len(trunc_w)))
            i += 1
            if i >= self.rows - 2:
                break
        while i < self.rows - 2:
            self.win_handler.addstr(i + 1, 1, ' ' * (self.cols - 2))
            i += 1
        self.win_handler.border()
        self.win_handler.refresh()
        if self.win_title is not None:
            self.win_title.refresh()

    def assign_lines(self, lines):
        self.lines = [w.word for w in lines if w.group == self.group]
        # print(self.lines)


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
                 level=logging.INFO):
    """Function to setup a generic loggers."""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--dry-run', action='store_false', dest='dry_run',
                        help='do not write the results on exit')
    return parser


def avg_or_zero(num, den):
    """Safely calculates an average, returning 0 if no elements are present."""
    if den > 0:
        avg = 100 * num / den
    else:
        avg = 0

    return avg


def get_stats_strings(words, related_items_count=0):
    stats_strings = []
    n_completed = words.count_classified()
    n_keywords = words.count_by_class(WordClass.KEYWORD)
    n_noise = words.count_by_class(WordClass.NOISE)
    n_not_relevant = words.count_by_class(WordClass.NOT_RELEVANT)
    n_later = words.count_by_class(WordClass.POSTPONED)
    stats_strings.append('Total words:  {:7}'.format(len(words.items)))
    avg = avg_or_zero(n_completed, len(words.items))
    stats_strings.append('Completed:    {:7} ({:6.2f}%)'.format(n_completed,
                                                                avg))
    avg = avg_or_zero(n_keywords, n_completed)
    stats_strings.append('Keywords:     {:7} ({:6.2f}%)'.format(n_keywords,
                                                                avg))
    avg = avg_or_zero(n_noise, n_completed)
    stats_strings.append('Noise:        {:7} ({:6.2f}%)'.format(n_noise, avg))
    avg = avg_or_zero(n_not_relevant, n_completed)
    stats_strings.append('Not relevant: {:7} ({:6.2f}%)'.format(n_not_relevant,
                                                                avg))
    avg = avg_or_zero(n_later, n_completed)
    stats_strings.append('Postponed:    {:7} ({:6.2f}%)'.format(n_later,
                                                                avg))
    s = 'Related:      {:7}'
    if related_items_count >= 0:
        s = s.format(related_items_count)
    else:
        s = s.format(0)

    stats_strings.append(s)
    return stats_strings


def find_word(string, substring):
    return any([substring == word for word in string.split()])
    # return substring in string


def init_curses():
    # create stdscr
    stdscr = curses.initscr()
    stdscr.clear()

    # allow echo, set colors
    curses.noecho()
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    return stdscr


def do_classify(klass, words, evaluated_word, sort_word_key,
                related_items_count, windows):
    windows[klass.classname].lines.append(evaluated_word)
    refresh_class_windows(evaluated_word, klass, windows)

    words.mark_word(evaluated_word, klass,
                    words.get_last_inserted_order() + 1, sort_word_key)

    if related_items_count <= 0:
        sort_word_key = evaluated_word

    containing, not_containing = words.return_related_items(sort_word_key)
    if related_items_count <= 0:
        related_items_count = len(containing) + 1

    windows['__WORDS'].lines = containing
    windows['__WORDS'].lines.extend(not_containing)
    windows['__WORDS'].display_lines(rev=False, highlight_word=sort_word_key)
    related_items_count -= 1
    return related_items_count, sort_word_key


def refresh_class_windows(evaluated_word, klass, windows):
    for win in windows:
        if win in ['__WORDS', '__STATS']:
            continue
        if win == klass.classname:
            windows[win].display_lines(rev=True, highlight_word=evaluated_word,
                                       color_pair=2)
        else:
            windows[win].display_lines(rev=True)


def undo(words, sort_word_key, related_items_count, windows, logger, profiler):
    last_word = words.get_last_inserted_word()
    if last_word is None:
        return related_items_count, sort_word_key

    group = last_word.group
    related = last_word.related
    logger.debug("Undo: {} group {} order {}".format(last_word.word,
                                                     group,
                                                     last_word.order))
    # un-mark last_word
    words.mark_word(last_word.word, WordClass.NONE, None)
    # remove last_word from the window that actually contains it
    try:
        win = windows[group.classname]
        win.lines.remove(last_word.word)
        prev_last_word = words.get_last_inserted_word()
        if prev_last_word is not None:
            refresh_class_windows(prev_last_word.word, prev_last_word.group,
                                  windows)
        else:
            refresh_class_windows('', WordClass.NONE, windows)
    except KeyError:
        pass  # if here the word is not in a window so nothing to do

    # handle related word
    if related == sort_word_key:
        related_items_count += 1
        rwl = [last_word.word]
        rwl.extend(windows['__WORDS'].lines)
        windows['__WORDS'].lines = rwl
    else:
        sort_word_key = related
        containing, not_containing = words.return_related_items(sort_word_key)
        windows['__WORDS'].lines = containing
        windows['__WORDS'].lines.extend(not_containing)
        windows['__WORDS'].display_lines(rev=False,
                                         highlight_word=sort_word_key)
        related_items_count = len(containing) + 1

    if sort_word_key == '':
        # if sort_word_key is empty there's no related item: fix the
        # related_items_count to the correct value of 0
        related_items_count = 0

    profiler.info("WORD '{}' UNDONE".format(last_word.word))

    return related_items_count, sort_word_key


def create_windows(win_width, rows):
    windows = dict()
    win_classes = [WordClass.KEYWORD, WordClass.RELEVANT, WordClass.NOISE,
                   WordClass.NOT_RELEVANT, WordClass.POSTPONED]
    for i, cls in enumerate(win_classes):
        windows[cls.classname] = Win(cls, title=cls.classname.capitalize(),
                                     rows=rows, cols=win_width, y=(rows + 1) * i,
                                     x=0, show_title=True)

    windows['__WORDS'] = Win(None, rows=27, cols=win_width, y=9, x=win_width)
    windows['__STATS'] = Win(None, rows=9, cols=win_width, y=0, x=win_width)
    return windows


def curses_main(scr, words, datafile, logger=None, profiler=None):
    stdscr = init_curses()
    win_width = 40
    rows = 8

    # define windows
    windows = create_windows(win_width, rows)

    curses.ungetch(' ')
    _ = stdscr.getch()
    for win in windows:
        if win in ['__WORDS', '__STATS']:
            continue

        windows[win].assign_lines(words.items)
        windows[win].display_lines()

    last_word = words.get_last_inserted_word()
    if last_word is None:
        related_items_count = 0
        sort_word_key = ''
        lines = [w.word for w in words.items if not w.is_grouped()]
    else:
        sort_word_key = last_word.related
        containing, not_containing = words.return_related_items(sort_word_key)
        related_items_count = len(containing)
        lines = containing
        lines.extend(not_containing)

    windows['__WORDS'].lines = lines
    windows['__STATS'].lines = get_stats_strings(words, related_items_count)
    windows['__STATS'].display_lines(rev=False)
    while True:
        if len(windows['__WORDS'].lines) <= 0:
            break
        evaluated_word = windows['__WORDS'].lines[0]
        if related_items_count <= 0:
            sort_word_key = ''

        windows['__WORDS'].display_lines(rev=False, highlight_word=sort_word_key)
        c = chr(stdscr.getch())
        classifing_keys = [WordClass.KEYWORD.key,
                           WordClass.NOT_RELEVANT.key,
                           WordClass.NOISE.key,
                           WordClass.RELEVANT.key]
        if c in classifing_keys:
            klass = WordClass.get_from_key(c)
            profiler.info("WORD '{}' AS '{}'".format(evaluated_word,
                                                     klass.classname))
            related_items_count, sort_word_key = do_classify(klass, words,
                                                             evaluated_word,
                                                             sort_word_key,
                                                             related_items_count,
                                                             windows)
        elif c == 'p':
            profiler.info("WORD '{}' POSTPONED".format(evaluated_word))
            # classification: POSTPONED
            words.mark_word(evaluated_word, WordClass.POSTPONED,
                            words.get_last_inserted_order() + 1,
                            sort_word_key)
            windows['__WORDS'].lines = windows['__WORDS'].lines[1:]
            windows[WordClass.POSTPONED.classname].lines.append(evaluated_word)
            refresh_class_windows(evaluated_word, WordClass.POSTPONED, windows)
            related_items_count -= 1
        elif c == 'w':
            # write to file
            words.to_csv(datafile)
        elif c == 'u':
            # undo last operation
            related_items_count, sort_word_key = undo(words, sort_word_key,
                                                      related_items_count,
                                                      windows, logger, profiler)

        elif c == 'q':
            # quit
            break
        windows['__STATS'].lines = get_stats_strings(words, related_items_count)
        windows['__STATS'].display_lines(rev=False)


def main():
    parser = init_argparser()
    args = parser.parse_args()

    profiler_logger = setup_logger('profiler_logger', 'profiler.log')
    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)

    profiler_logger.info("*** PROGRAM STARTED ***".format(args.datafile))
    profiler_logger.info("DATAFILE: '{}'".format(args.datafile))
    words = WordList()
    _, _ = words.from_csv(args.datafile)
    profiler_logger.info("CLASSIFIED: {}".format(words.count_classified()))
    curses.wrapper(curses_main, words, args.datafile, logger=debug_logger,
                   profiler=profiler_logger)
    profiler_logger.info("CLASSIFIED: {}".format(words.count_classified()))
    profiler_logger.info("DATAFILE '{}'".format(args.datafile))
    profiler_logger.info("*** PROGRAM TERMINATED ***")
    curses.endwin()

    if args.dry_run:
        words.to_csv(args.datafile)


if __name__ == "__main__":
    main()
