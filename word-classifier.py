import sys
import csv
import curses
import logging
import argparse
import operator
from dataclasses import dataclass


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


def load_words(infile):
    with open(infile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
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
                item = Word(
                    index=None,
                    word=row[header.index('keyword')],
                    count=row[header.index('count')],
                    group=row[header.index('group')],
                    order=order
                )
                items.append(item)
                line_count += 1
    return (header, items)


def write_words(outfile):
    with open(outfile, mode='w') as out:
        writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for w in words:
            item = [w.word, w.count, w.group, w.order]
            writer.writerow(item)


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

    def display_lines(self, rev=True, highlight_word=None):
        if rev:
            word_list = reversed(self.lines)
        else:
            word_list = self.lines
        i = 0
        for w in word_list:
            trunc_w = w[:self.cols - 2]
            if highlight_word is None:
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


def get_stats_strings(words):
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
    return stats_strings


def find_word(string, substring):
   return substring in string


def return_related_items(words, key):
    containing = []
    not_containing = []
    for w in words:
        if (w.group != ''):
            continue
        if (find_word(w.word, key)):
            containing.append(w.word)
        else:
            not_containing.append(w.word)
    return containing, not_containing


def mark_word(words, word, marker, order):
    for w in words:
        if w.word == word:
            w.group = marker
            w.order = order
            break
    return words


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


def get_last_inserted_order(words):
    orders = [w.order for w in words if w.order is not None]
    if len(orders) == 0:
        order = 0
    else:
        order = max(orders)
    return order


def main(args, words, datafile):
    stdscr = init_curses()
    win_width = 40

    # define windows
    windows = {
        KEYWORD: Win(keys[KEYWORD], title='Keywords', rows=12, cols=win_width, y=0, x=0),
        NOISE: Win(keys[NOISE], title='Noise', rows=12, cols=win_width, y=12, x=0),
        NOTRELEVANT: Win(keys[NOTRELEVANT], title='Not-relevant', rows=12, cols=win_width, y=24, x=0)
    }
    curses.ungetch(' ')
    c = stdscr.getch()
    for win in windows:
        windows[win].assign_lines(words)
        windows[win].display_lines()
    words_window = Win(None, rows=27, cols=win_width, y=9, x=win_width)
    stats_window = Win(None, rows=9, cols=win_width, y=0, x=win_width)
    stats_window.lines = get_stats_strings(words)
    stats_window.display_lines(rev=False)

    related_items_count = 0
    words_window.lines = [w.word for w in words if not w.is_grouped()]
    sort_word_key = None
    order = get_last_inserted_order(words)
    while True:
        if len(words_window.lines) <= 0:
            break
        evaluated_word = words_window.lines[0]
        words_window.display_lines(rev=False, highlight_word=sort_word_key)
        c = stdscr.getch()
        #if c in [ord(keys[KEYWORD]), ord(keys[NOTRELEVANT])]:
        #    # classification: KEYWORD or NOTRELEVANT
        #    words = mark_word(words, evaluated_word, chr(c), order)
        #    order += 1
        #    win = windows[key2class[chr(c)]]
        #    win.lines.append(evaluated_word)
        #    words_window.lines = words_window.lines[1:]
        #    win.display_lines()
        #    related_items_count -= 1
        if c in [ord(keys[KEYWORD]), ord(keys[NOTRELEVANT]), ord(keys[NOISE])]:
        #elif c in [ord(keys[NOISE])]:
            # classification: KEYWORD, NOTRELEVANT or NOISE
            words = mark_word(words, evaluated_word, chr(c), order)
            order += 1
            win = windows[key2class[chr(c)]]
            win.lines.append(evaluated_word)
            win.display_lines()
            if related_items_count <= 0:
                sort_word_key = evaluated_word
            containing, not_containing = return_related_items(words, sort_word_key)
            if related_items_count <= 0:
                related_items_count = len(containing) + 1
            #logging.debug("sort_word_key: {}".format(sort_word_key))
            #logging.debug("related_items_count: {}".format(related_items_count))
            words_window.lines = containing
            words_window.lines.extend(not_containing)
            #logging.debug("containing: {}".format(containing))
            #logging.debug("words_window.lines: {}".format(words_window.lines))
            words_window.display_lines(rev=False, highlight_word=sort_word_key)
            related_items_count -= 1
        elif c == ord('p'):
            # classification: POSTPONED
            words = mark_word(words, evaluated_word, chr(c), order)
            order += 1
            words_window.lines = words_window.lines[1:]
            related_items_count -= 1
        elif c == ord('w'):
            # write to file
            write_words(datafile)
        elif c == ord('u'):
            # undo last operation
            orders = [w.order for w in words if w.order is not None]
            if len(orders) == 0:
                continue
            else:
                max_index, max_value = max(enumerate(orders), key=operator.itemgetter(1))
            w = words[max_index].word
            logging.debug("{} {} {}".format(max_index, max_value, w))
            words = mark_word(words, w, '', '')
            order -= 1
            rwl = [w]
            rwl.extend(words_window.lines)
            words_window.lines = rwl
        elif c == ord('q'):
            # quit
            break
        stats_window.lines = get_stats_strings(words)
        stats_window.lines.append('Related:      {:7}'.format(related_items_count if related_items_count >= 0 else 0))
        stats_window.display_lines(rev=False)


if __name__ == "__main__":
    logging.basicConfig(filename='slr-kit.log',
            filemode='a',
            format='%(asctime)s [%(levelname)s] %(message)s',
            level=logging.DEBUG)
    parser = init_argparser()
    args = parser.parse_args()

    logging.debug("**************** PROGRAM STARTED ****************")
    (header, words) = load_words(args.datafile)

    curses.wrapper(main, words, args.datafile)
    curses.endwin()

    if args.dry_run:
        write_words(args.datafile)
