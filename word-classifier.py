import argparse
import csv
import curses
import enum
import logging
from dataclasses import dataclass


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
                 level=logging.INFO):
    """Function to setup a generic loggers."""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


profiler_logger = setup_logger('profiler_logger', 'profiler.log')
debug_logger = setup_logger('debug_logger', 'slr-kit.log', level=logging.DEBUG)


# List of class names
class ClassNames(enum.Enum):
    NONE = ('', '')
    KEYWORD = ('keyword', 'k')
    NOISE = ('noise', 'n')
    RELEVANT = ('relevant', 'r')
    NOT_RELEVANT = ('not-relevant', 'x')
    POSTPONED = ('postponed', 'p')

    @classmethod
    def key2class(cls, key):
        for clname in cls:
            if clname.key == key:
                return clname

        raise ValueError('"{}" is not a valid key'.format(key))

    def __init__(self, classname, key):
        self.classname = classname
        self.key = key


@dataclass
class Word:
    index: int
    word: str
    count: int
    group: str
    order: int
    related: str

    def is_grouped(self):
        return self.group != ''


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
                item = Word(
                    index=0,
                    word=row['keyword'],
                    count=row['count'],
                    group=row['group'],
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
                        'group': w.group,
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
            if w.group != '':
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

    def __init__(self, key, title='', rows=3, cols=30, y=0, x=0):
        self.key = key
        self.title = title
        self.rows = rows
        self.cols = cols
        self.y = y
        self.x = x
        self.win_handler = curses.newwin(self.rows, self.cols, self.y, self.x)
        self.win_handler.border()
        self.win_handler.refresh()
        self.lines = []

    def display_lines(self, rev=True, highlight_word=''):
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
                    self.win_handler.addstr(highlight_word, curses.color_pair(1))
                self.win_handler.addstr(i + 1, len(trunc_w) + 1, ' ' * (self.cols - 2 - len(trunc_w)))
            i += 1
            if i >= self.rows - 2:
                break
        while i < self.rows - 2:
            self.win_handler.addstr(i + 1, 1, ' ' * (self.cols - 2))
            i += 1
        self.win_handler.border()
        self.win_handler.refresh()

    def assign_lines(self, lines):
        self.lines = [w.word for w in lines if w.group == self.key]
        # print(self.lines)


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--dry-run', action='store_false', dest='dry_run',
                        help='do not write the results on exit')
    return parser


def avg_or_zero(num, den):
    """Safely calculatess an average, returning 0 if no elements are present."""
    if den > 0:
        avg = 100 * num / den
    else:
        avg = 0

    return avg


def get_stats_strings(words, related_items_count=0):
    stats_strings = []
    n_completed = words.count_classified()
    n_keywords = words.count_by_class('k')
    n_noise = words.count_by_class('n')
    n_not_relevant = words.count_by_class('x')
    n_later = words.count_by_class('p')
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

    return stdscr


def do_classify(key, words, evaluated_word, sort_word_key, related_items_count,
                windows):
    klass = ClassNames.key2class(key)
    win = windows[klass.classname]
    win.lines.append(evaluated_word)
    win.display_lines(rev=True)
    words.mark_word(evaluated_word, key,
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


def undo(words, sort_word_key, related_items_count, windows, logger, profiler):
    last_word = words.get_last_inserted_word()
    if last_word is None:
        return related_items_count, sort_word_key

    group = last_word.group
    related = last_word.related
    logger.debug("Undo: {} group {} order {}".format(last_word.word,
                                                     group,
                                                     last_word.order))
    # remove last_word from the window that actually contains it
    try:
        win = windows[ClassNames.key2class(group).classname]
        win.lines.remove(last_word.word)
        win.display_lines(rev=True)
    except KeyError:
        pass  # if here the word is not in a window so nothing to do

    # un-mark last_word
    words.mark_word(last_word.word, '', None)
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
        windows['__WORDS'].display_lines(rev=False, highlight_word=sort_word_key)
        related_items_count = len(containing) + 1

    if sort_word_key == '':
        # if sort_word_key is empty there's no related item: fix the
        # related_items_count to the correct value of 0
        related_items_count = 0

    profiler.info("WORD '{}' UNDONE".format(last_word.word))

    return related_items_count, sort_word_key


def curses_main(scr, words, datafile, logger=None, profiler=None):
    stdscr = init_curses()
    win_width = 40

    # define windows
    windows = {
        ClassNames.KEYWORD.classname: Win(ClassNames.KEYWORD.key,
                                          title='Keywords', rows=8,
                                          cols=win_width, y=0, x=0),
        ClassNames.RELEVANT.classname: Win(ClassNames.RELEVANT.key,
                                           title='Relevant', rows=8,
                                           cols=win_width, y=8, x=0),
        ClassNames.NOISE.classname: Win(ClassNames.NOISE.key, title='Noise',
                                        rows=8, cols=win_width, y=16, x=0),
        ClassNames.NOT_RELEVANT.classname: Win(ClassNames.NOT_RELEVANT.key,
                                               title='Not-relevant', rows=8,
                                               cols=win_width, y=24, x=0),
        '__WORDS': Win(None, rows=27, cols=win_width, y=9, x=win_width),
        '__STATS': Win(None, rows=9, cols=win_width, y=0, x=win_width)
    }

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
        classifing_keys = [ClassNames.KEYWORD.key,
                           ClassNames.NOT_RELEVANT.key,
                           ClassNames.NOISE.key,
                           ClassNames.RELEVANT.key]
        if c in classifing_keys:
            profiler.info("WORD '{}' AS '{}'".format(evaluated_word,
                                                     ClassNames.key2class(c)))
            related_items_count, sort_word_key = do_classify(c, words,
                                                             evaluated_word,
                                                             sort_word_key,
                                                             related_items_count,
                                                             windows)
        elif c == 'p':
            profiler.info("WORD '{}' POSTPONED".format(evaluated_word))
            # classification: POSTPONED
            words.mark_word(evaluated_word, c,
                            words.get_last_inserted_order() + 1,
                            sort_word_key)
            windows['__WORDS'].lines = windows['__WORDS'].lines[1:]
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
