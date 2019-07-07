import sys
import csv
import curses
import logging
import argparse


def init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
        help="input CSV data file")
    parser.add_argument('--dry-run', action='store_false', dest='dry_run',
        help='do not write the reults on exit')
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
                items.append(row)
                line_count += 1
    # appending incremental number to keep track of the original ordering
    items2 = []
    for i, x in enumerate(items):
        x.append(i)
        items2.append(x)
    return (header, items2)


def write_words(outfile):
    with open(outfile, mode='w') as out:
        writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for w in words:
            writer.writerow(w[:3])


class Win(object):
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
        self.words = []

    def display_words(self, rev=True, highlight_word=None):
        if rev:
            word_list = reversed(self.words)
        else:
            word_list = self.words
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

    def assign_words(self, words):
        self.words = [w[0] for w in words if w[2] == self.key]
        #print(self.words)


def avg_or_zero(num, den):
    if den > 0:
        avg = 100 * num / den
    else:
        avg = 0
    return avg


def get_stats_strings(words):
    stats_strings = []
    n_completed = len([w for w in words if w[2] != ''])
    n_keywords = len([w for w in words if w[2] == 'k'])
    n_noise = len([w for w in words if w[2] == 'n'])
    n_not_relevant = len([w for w in words if w[2] == 'x'])
    n_later = len([w for w in words if w[2] == 'l'])
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


def sort_words(words, word):
    #logging.debug("word: {}".format(word))
    containing = [w[0] for w in words if (word in w[0]) and (w[2] == '') and (' ' in w[0])]
    #logging.debug("containing: {}".format(containing))
    not_containing = [w[0] for w in words if (not word in w[0]) and (w[2] == '')]
    #logging.debug("not containing: {}".format(not_containing))
    containing.extend(not_containing)
    #logging.debug("containing: {}".format(containing))
    return containing


def mark_word(words, word, marker):
    for w in words:
        if w[0] == word:
            w[2] = marker
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


def main(args, words):
    stdscr = init_curses()
    win_width = 40

    # define windows
    windows = {
        'k': Win('k', title='Keywords', rows=12, cols=win_width, y=0, x=0),
        'n': Win('n', title='Noise', rows=12, cols=win_width, y=12, x=0),
        'x': Win('x', title='Not-relevant', rows=12, cols=win_width, y=24, x=0)
    }
    curses.ungetch(' ')
    c = stdscr.getch()
    for win in windows:
        windows[win].assign_words(words)
        windows[win].display_words()
    words_window = Win(None, rows=27, cols=win_width, y=9, x=win_width)
    stats_window = Win(None, rows=9, cols=win_width, y=0, x=win_width)
    stats_window.words = get_stats_strings(words)
    stats_window.display_words(rev=False)

    related_items_count = 0
    words_window.words = [w[0] for w in words if w[2] == '']
    sort_word_key = None
    while True:
        evaluated_word = words_window.words[0]
        words_window.display_words(rev=False, highlight_word=sort_word_key)
        c = stdscr.getch()
        if c in [ord('k'), ord('x')]:
            words = mark_word(words, evaluated_word, chr(c))
            win = windows[chr(c)]
            win.words.append(evaluated_word)
            words_window.words = words_window.words[1:]
            win.display_words()
            related_items_count -= 1
        elif c == ord('n'):
            words = mark_word(words, evaluated_word, chr(c))
            win = windows[chr(c)]
            win.words.append(evaluated_word)
            win.display_words()
            if related_items_count <= 0:
                sort_word_key = evaluated_word
                related_items_count = len([w for w in words if (sort_word_key in w[0]) and (w[2] == '') and (' ' in w[0])]) + 1
            #logging.debug("related_items_count: {}".format(related_items_count))
            words_window.words = sort_words(words, sort_word_key)
            #logging.debug("words_window.words: {}".format(words_window.words))
            words_window.display_words(rev=False, highlight_word=sort_word_key)
            related_items_count -= 1
        elif c == ord('l'):
            words = mark_word(words, evaluated_word, chr(c))
            words_window.words = words_window.words[1:]
            related_items_count -= 1
        elif c == ord('q'):
            break
        stats_window.words = get_stats_strings(words)
        stats_window.words.append('Related:      {:7}'.format(related_items_count if related_items_count >= 0 else 0))
        stats_window.display_words(rev=False)


if __name__ == "__main__":
    logging.basicConfig(filename='slr-kit.log',
            filemode='a',
            format='%(asctime)s [%(levelname)s] %(message)s',
            level=logging.DEBUG)
    parser = init_argparser()
    args = parser.parse_args()

    (header, words) = load_words(args.datafile)

    curses.wrapper(main, words)
    curses.endwin()

    if args.dry_run:
        write_words(args.datafile)
