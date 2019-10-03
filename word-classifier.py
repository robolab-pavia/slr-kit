import sys
import csv
import curses
import logging
import argparse
import operator
from dataclasses import dataclass


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'), level=logging.INFO):
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
KEYWORD = 'keyword'
NOISE = 'noise'
RELEVANT = 'relevant'
NOTRELEVANT = 'not-relevant'

keys = {
    KEYWORD: 'k',
    NOISE: 'n',
    RELEVANT:'r',
    NOTRELEVANT: 'x'
}


# FIXME: the key2class dict shall be obtained by "reversing" the keys dict
key2class = {
    'k': KEYWORD,
    'n': NOISE,
    'r': RELEVANT,
    'x': NOTRELEVANT
}


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


def init_argparser():
    """Initialize the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
        help="input CSV data file")
    parser.add_argument('--dry-run', action='store_false', dest='dry_run',
        help='do not write the results on exit')
    return parser


class WordList(object):
    def __init__(self, items=None):
        self.items = items
        self.csv_header = None

    def from_csv(self, infile):
        with open(infile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = None
            line_count = 0
            items = []
            for row in csv_reader:
                if line_count == 0:
                    header = row
                    line_count += 1
                else:
                    order_value = row[header.index('order')]
                    if order_value == '':
                        order = None
                    else:
                        order = int(order_value)

                    if 'related' in header:
                        related = row[header.index('related')]
                    else:
                        related = ''

                    item = Word(
                        index=0,
                        word=row[header.index('keyword')],
                        count=row[header.index('count')],
                        group=row[header.index('group')],
                        order=order,
                        related=related
                    )
                    items.append(item)
                    line_count += 1

        if 'related' not in header:
            header.append('related')

        self.csv_header = header
        self.items = items
        return header, items

    def to_csv(self, outfile):
        with open(outfile, mode='w') as out:
            writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.csv_header)
            for w in self.items:
                # FIXME: this ordering should depend from the header
                item = [w.word, w.count, w.group, w.order, w.related]
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
        last_word = None
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
            if (w.group != ''):
                continue
            if (find_word(w.word, key)):
                containing.append(w.word)
            else:
                not_containing.append(w.word)
        return containing, not_containing


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
                self.win_handler.addstr(i + 1, 1, trunc_w + ' '*(self.cols - 2 - len(trunc_w)))
            else:
                tok = w.split(highlight_word)
                self.win_handler.addstr(i + 1, 1, '')
                for t in tok:
                    self.win_handler.addstr(t)
                    self.win_handler.addstr(highlight_word, curses.color_pair(1))
                self.win_handler.addstr(i + 1, len(trunc_w) + 1, ' '*(self.cols - 2 - len(trunc_w)))
            i += 1
            if i >= self.rows - 2:
                break
        while i < self.rows - 2:
            self.win_handler.addstr(i + 1, 1, ' '*(self.cols - 2))
            i += 1
        self.win_handler.border()
        self.win_handler.refresh()

    def assign_lines(self, lines):
        self.lines = [w.word for w in lines if w.group == self.key]
        #print(self.lines)


def avg_or_zero(num, den):
    """Safely calculatess an average, returning 0 if no elements are present."""
    if den > 0:
        avg = 100 * num / den
    else:
        avg = 0
    return avg


def get_stats_strings(words, related_items_count=0):
    stats_strings = []
    n_completed = len([w for w in words if w.is_grouped()])
    n_keywords = len([w for w in words if w.group == 'k'])
    n_noise = len([w for w in words if w.group == 'n'])
    n_not_relevant = len([w for w in words if w.group == 'x'])
    n_later = len([w for w in words if w.group == 'p'])
    stats_strings.append('Total words:  {:7}'.format(len(words)))
    avg = avg_or_zero(n_completed, len(words))
    stats_strings.append('Completed:    {:7} ({:6.2f}%)'.format(n_completed, avg))
    avg = avg_or_zero(n_keywords, n_completed)
    stats_strings.append('Keywords:     {:7} ({:6.2f}%)'.format(n_keywords, avg))
    avg = avg_or_zero(n_noise, n_completed)
    stats_strings.append('Noise:        {:7} ({:6.2f}%)'.format(n_noise, avg))
    avg = avg_or_zero(n_not_relevant, n_completed)
    stats_strings.append('Not relevant: {:7} ({:6.2f}%)'.format(n_not_relevant, avg))
    avg = avg_or_zero(n_later, n_completed)
    stats_strings.append('Postponed:    {:7} ({:6.2f}%)'.format(n_later, avg))
    stats_strings.append('Related:      {:7}'.format(related_items_count if related_items_count >= 0 else 0))
    return stats_strings


def find_word(string, substring):
    return any([substring == word for word in string.split()])
    return substring in string


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


def main(args, words, datafile, logger=None, profiler=None):
    stdscr = init_curses()
    win_width = 40

    # define windows
    windows = {
        KEYWORD: Win(keys[KEYWORD], title='Keywords', rows=8, cols=win_width, y=0, x=0),
        RELEVANT: Win(keys[RELEVANT], title='Relevant', rows=8, cols=win_width, y=8, x=0),
        NOISE: Win(keys[NOISE], title='Noise', rows=8, cols=win_width, y=16, x=0),
        NOTRELEVANT: Win(keys[NOTRELEVANT], title='Not-relevant', rows=8, cols=win_width, y=24, x=0)
    }
    curses.ungetch(' ')
    c = stdscr.getch()
    for win in windows:
        windows[win].assign_lines(words.items)
        windows[win].display_lines()
    words_window = Win(None, rows=27, cols=win_width, y=9, x=win_width)

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

    words_window.lines = lines
    stats_window = Win(None, rows=9, cols=win_width, y=0, x=win_width)
    stats_window.lines = get_stats_strings(words.items, related_items_count)
    stats_window.display_lines(rev=False)
    while True:
        if len(words_window.lines) <= 0:
            break
        evaluated_word = words_window.lines[0]
        if related_items_count <= 0:
            sort_word_key = ''

        words_window.display_lines(rev=False, highlight_word=sort_word_key)
        c = stdscr.getch()
        #if c in [ord(keys[KEYWORD]), ord(keys[NOTRELEVANT])]:
        #    # classification: KEYWORD or NOTRELEVANT
        #    words.mark_word(evaluated_word, chr(c), order)
        #    win = windows[key2class[chr(c)]]
        #    win.lines.append(evaluated_word)
        #    words_window.lines = words_window.lines[1:]
        #    win.display_lines()
        #    related_items_count -= 1
        if c in [ord(keys[KEYWORD]), ord(keys[NOTRELEVANT]), ord(keys[NOISE]), ord(keys[RELEVANT])]:
        #elif c in [ord(keys[NOISE])]:
            # classification: KEYWORD, RELEVANT, NOTRELEVANT or NOISE
            profiler.info("WORD '{}' AS '{}'".format(evaluated_word, key2class[chr(c)]))
            win = windows[key2class[chr(c)]]
            win.lines.append(evaluated_word)
            win.display_lines()
            words.mark_word(evaluated_word, chr(c),
                            words.get_last_inserted_order() + 1, sort_word_key)
            if related_items_count <= 0:
                sort_word_key = evaluated_word

            containing, not_containing = words.return_related_items(sort_word_key)
            if related_items_count <= 0:
                related_items_count = len(containing) + 1
            #logger.debug("sort_word_key: {}".format(sort_word_key))
            #logger.debug("related_items_count: {}".format(related_items_count))
            words_window.lines = containing
            words_window.lines.extend(not_containing)
            #logger.debug("containing: {}".format(containing))
            #logger.debug("words_window.lines: {}".format(words_window.lines))
            words_window.display_lines(rev=False, highlight_word=sort_word_key)
            related_items_count -= 1
        elif c == ord('p'):
            # classification: POSTPONED
            words.mark_word(evaluated_word, chr(c),
                            words.get_last_inserted_order() + 1,
                            sort_word_key)
            words_window.lines = words_window.lines[1:]
            related_items_count -= 1
        elif c == ord('w'):
            # write to file
            words.to_csv(datafile)
        elif c == ord('u'):
            # undo last operation
            last_word = words.get_last_inserted_word()

            if last_word is None:
                continue

            group = last_word.group
            related = last_word.related
            logger.debug("Undo: {} group {} order {}".format(last_word.word,
                                                             group,
                                                             last_word.order))
            # remove last_word from the window that actually contains it
            try:
                win = windows[key2class[group]]
                win.lines.remove(last_word.word)
                win.display_lines(rev=False)
            except KeyError:
                pass  # if here the word is not in a window so nothing to do

            # un-mark last_word
            words.mark_word(last_word.word, '', None)
            if related == sort_word_key:
                related_items_count += 1
                rwl = [last_word.word]
                rwl.extend(words_window.lines)
                words_window.lines = rwl
            else:
                sort_word_key = related
                containing, not_containing = words.return_related_items(sort_word_key)
                words_window.lines = containing
                words_window.lines.extend(not_containing)
                words_window.display_lines(rev=False, highlight_word=sort_word_key)
                related_items_count = len(containing) + 1

            if sort_word_key == '':
                # if sort_word_key is empty there's no related item: fix the
                # related_items_count to the correct value of 0
                related_items_count = 0

        elif c == ord('q'):
            # quit
            break
        stats_window.lines = get_stats_strings(words.items, related_items_count)
        stats_window.display_lines(rev=False)


if __name__ == "__main__":
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    parser = init_argparser()
    args = parser.parse_args()

    profiler_logger.info("*** PROGRAM STARTED ***")
    words = WordList()
    (header, items) = words.from_csv(args.datafile)

    curses.wrapper(main, words, args.datafile, logger=debug_logger, profiler=profiler_logger)
    profiler_logger.info("*** PROGRAM TERMINATED ***")
    curses.endwin()

    if args.dry_run:
        words.to_csv(args.datafile)
