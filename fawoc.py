import argparse
import curses
import json
import logging
import os
import pathlib
import sys
from terms import Label, TermList


class Win(object):
    """
    Contains the list of lines to display.

    :type label: Label or None
    :type title: str
    :type rows: int
    :type cols: int
    :type y: int
    :type x: int
    :type win_title: _curses.window or None
    :type win_handler: _curses.window
    :type lines: list[str]
    """

    def __init__(self, label, title='', rows=3, cols=30, y=0, x=0,
                 show_title=False):
        """
        Creates a window

        :param label: label associated to the windows
        :type label: Label or None
        :param title: title of window. Default: empty string
        :type title: str
        :param rows: number of rows
        :type rows: int
        :param cols: number of columns
        :type cols: int
        :param y: y coordinate
        :type y: int
        :param x: x coordinate
        :type x: int
        :param show_title: if True the window must show its title. Default: False
        :type show_title: bool
        """
        self.label = label
        self.title = title
        self.rows = rows
        self.cols = cols
        self.x = x
        if show_title:
            self.y = y + 1
            self.win_title = curses.newwin(1, self.cols, y, self.x)
            self.win_title.addstr(' {}'.format(self.title))
        else:
            self.y = y
            self.win_title = None

        self.win_handler = curses.newwin(self.rows, self.cols, self.y, self.x)
        self.win_handler.border()
        self.win_handler.refresh()
        self.lines = []

    def display_lines(self, rev=True, highlight_word='', only_the_word=False,
                      color_pair=1):
        """
        Display the lines associated to the window

        :param rev: if True, display lines in reversed order. Default: True
        :type rev: bool
        :param highlight_word: the word to highlight. Default: empty string
        :type highlight_word: str
        :param only_the_word: if True only highlight_word is highlighted.
        :type only_the_word: bool
        :param color_pair: the curses color pair to use to hightlight. Default 1
        :type color_pair: int
        """
        if rev:
            word_list = reversed(self.lines)
        else:
            word_list = self.lines

        i = 0
        for w in word_list:
            if i >= self.rows - 2:
                break

            self._display_line(w, highlight_word, only_the_word, i, color_pair)
            i += 1

        while i < self.rows - 2:
            self.win_handler.addstr(i + 1, 1, ' ' * (self.cols - 2))
            i += 1

        self.win_handler.border()
        self.win_handler.refresh()
        if self.win_title is not None:
            self.win_title.refresh()

    def _display_line(self, line, highlight_word, only_word, line_index,
                      color_pair):
        """
        Display a single line in a window taking care of the word highlighting

        :param line: the line to display
        :type line: str
        :param highlight_word: the word to highlight
        :type highlight_word: str
        :param only_word: if True, highlight only highlight_word
        :type only_word: bool
        :param line_index: index of the line to display
        :type line_index: int
        :param color_pair: color pair for highlight
        :type color_pair: int
        """
        trunc_w = line[:self.cols - 2]
        l_trunc_w = len(trunc_w)
        pad = ' ' * (self.cols - 2 - l_trunc_w)
        flag = line != highlight_word and only_word
        if highlight_word == '' or flag:
            self.win_handler.addstr(line_index + 1, 1, trunc_w + pad)
        elif line == highlight_word:
            self.win_handler.addstr(line_index + 1, 1, trunc_w + pad,
                                    curses.color_pair(color_pair))
        else:
            tok = line.split(highlight_word)
            tok_len = len(tok)
            if tok_len == 1:
                # no highlight_word found
                self.win_handler.addstr(line_index + 1, 1, trunc_w + pad)
            else:
                self.win_handler.addstr(line_index + 1, 1, '')
                for i, t in enumerate(tok):
                    self.win_handler.addstr(t)
                    if i < tok_len - 1:
                        self.win_handler.addstr(highlight_word,
                                                curses.color_pair(color_pair))

                self.win_handler.addstr(line_index + 1, l_trunc_w + 1, pad)

    def assign_lines(self, terms):
        """
        Assign the terms in terms with the same label as the window
        :param terms: the terms list
        :type terms: list[Term]
        """
        terms = sorted(terms, key=lambda t: t.order)
        self.lines = [w.term for w in terms if w.label == self.label]


def setup_logger(name, log_file, formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
                 level=logging.INFO):
    """
    Function to setup a generic loggers.

    :param name: name of the logger
    :type name: str
    :param log_file: file of the log
    :type log_file: str
    :param formatter: formatter to be used by the logger
    :type formatter: logging.Formatter
    :param level: level to display
    :type level: int
    :return: the logger
    :rtype: logging.Logger
    """
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="input CSV data file")
    parser.add_argument('--input', '-i', metavar='LABEL',
                        help='input only the terms classified with the specified label')
    parser.add_argument('--dry-run', action='store_true', dest='dry_run',
                        help='do not write the results on exit')
    parser.add_argument('--no-auto-save', action='store_true', dest='no_auto_save',
                        help='disable auto-saving; save changes at the end of the session')
    parser.add_argument('--no-profile', action='store_true', dest='no_profile',
                        help='disable profiling logging')
    return parser


def avg_or_zero(num, den):
    """
    Safely calculates an average, returning 0 if no elements are present.

    :param num: numerator
    :type num: int
    :param den: denominator
    :type den: int
    """
    if den > 0:
        avg = 100 * num / den
    else:
        avg = 0

    return avg


def get_stats_strings(terms, related_items_count=0):
    """
    Calculates the statistics and formats them into strings

    :param terms: the list of terms
    :type terms: TermList
    :param related_items_count: the current number of related term
    :type related_items_count: int
    :return: the statistics about words formatted as strings
    :rtype: list[str]
    """
    stats_strings = []
    n_completed = terms.count_classified()
    n_keywords = terms.count_by_label(Label.KEYWORD)
    n_noise = terms.count_by_label(Label.NOISE)
    n_not_relevant = terms.count_by_label(Label.NOT_RELEVANT)
    n_later = terms.count_by_label(Label.POSTPONED)
    stats_strings.append('Total words:  {:7}'.format(len(terms.items)))
    avg = avg_or_zero(n_completed, len(terms.items))
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


def init_curses():
    """
    Initialize curses

    :return: the screen object
    :rtype: _curses.window
    """
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


def do_classify(label, terms, review, evaluated_term, sort_word_key,
                related_items_count, windows):
    """
    Handle the term classification process of the evaluated_term

    :param label: label to be assigned to the evaluated_term
    :type label: Label
    :param terms: the list of terms
    :type terms: TermList
    :param review: label under review
    :type review: Label
    :param evaluated_term: term to classify
    :type evaluated_term: str
    :param sort_word_key: actual term used for the related terms
    :type sort_word_key: str
    :param related_items_count: actual number of related terms
    :type related_items_count: int
    :param windows: dict of the windows
    :type windows: dict[str, Win]
    :return: the new sort_word_key and the new number of related terms
    :rtype: (str, int)
    """
    windows[label.label_name].lines.append(evaluated_term)
    refresh_label_windows(evaluated_term, label, windows)

    terms.classify_term(evaluated_term, label,
                        terms.get_last_classified_order() + 1, sort_word_key)

    if related_items_count <= 0:
        sort_word_key = evaluated_term

    containing, not_containing = terms.return_related_items(sort_word_key,
                                                            label=review)

    if related_items_count <= 0:
        related_items_count = len(containing) + 1

    windows['__WORDS'].lines = containing
    windows['__WORDS'].lines.extend(not_containing)
    windows['__WORDS'].display_lines(rev=False, highlight_word=sort_word_key)
    related_items_count -= 1
    return related_items_count, sort_word_key


def refresh_label_windows(term_to_highlight, label, windows):
    """
    Refresh the windows associated with a label

    :param term_to_highlight: the term to highlight
    :type term_to_highlight: str
    :param label: label of the window that has to highlight the term
    :type label: Label
    :param windows: dict of the windows
    :type windows: dict[str, Win]
    """
    for win in windows:
        if win in ['__WORDS', '__STATS']:
            continue
        if win == label.label_name:
            windows[win].display_lines(rev=True,
                                       highlight_word=term_to_highlight,
                                       only_the_word=True,
                                       color_pair=2)
        else:
            windows[win].display_lines(rev=True)


def undo(terms, review, sort_word_key, related_items_count, windows, logger,
         profiler):
    """
    Handle the undo of a term

    :param terms: the list of terms
    :type terms: TermList
    :param review: label under review
    :type review: Label
    :param sort_word_key: actual term used for the related terms
    :type sort_word_key: str
    :param related_items_count: actual number of related terms
    :type related_items_count: int
    :param windows: dict of the windows
    :type windows: dict[str, Win]
    :param logger: debug logger
    :type logger: logging.Logger
    :param profiler: profiling logger
    :type profiler: logging.Logger
    :return: the new sort_word_key and the new number of related terms
    :rtype: (str, int)
    """
    last_word = terms.get_last_classified_term()
    if last_word is None:
        return related_items_count, sort_word_key

    group = last_word.label
    related = last_word.related
    logger.debug("Undo: {} group {} order {}".format(last_word.term,
                                                     group,
                                                     last_word.order))
    # un-mark last_word
    terms.classify_term(last_word.term, review, -1)
    # remove last_word from the window that actually contains it
    try:
        win = windows[group.label_name]
        win.lines.remove(last_word.term)
        prev_last_word = terms.get_last_classified_term()
        if prev_last_word is not None:
            refresh_label_windows(prev_last_word.term, prev_last_word.label,
                                  windows)
        else:
            refresh_label_windows('', Label.NONE, windows)
    except KeyError:
        pass  # if here the word is not in a window so nothing to do

    # handle related word
    if related == sort_word_key:
        related_items_count += 1
        rwl = [last_word.term]
        rwl.extend(windows['__WORDS'].lines)
        windows['__WORDS'].lines = rwl
    else:
        sort_word_key = related
        containing, not_containing = terms.return_related_items(sort_word_key,
                                                                label=review)
        related_items_count = len(containing)
        windows['__WORDS'].lines = containing
        windows['__WORDS'].lines.extend(not_containing)
        windows['__WORDS'].display_lines(rev=False,
                                         highlight_word=sort_word_key)

    if sort_word_key == '':
        # if sort_word_key is empty there's no related item: fix the
        # related_items_count to the correct value of 0
        related_items_count = 0

    profiler.info("WORD '{}' UNDONE".format(last_word.term))

    return related_items_count, sort_word_key


def create_windows(win_width, rows, review):
    """
    Creates all the windows

    :param win_width: number of columns of each windows
    :type win_width: int
    :param rows: number of row of each windows
    :type rows: int
    :param review: label to review
    :type review: Label
    :return: the dict of the windows
    :rtype: dict[str, _curses.window]
    """
    windows = dict()
    win_classes = [Label.KEYWORD, Label.RELEVANT, Label.NOISE,
                   Label.NOT_RELEVANT, Label.POSTPONED]
    for i, cls in enumerate(win_classes):
        windows[cls.label_name] = Win(cls, title=cls.label_name.capitalize(),
                                      rows=rows, cols=win_width, y=(rows + 1) * i,
                                      x=0, show_title=True)

    title = 'Input label: {}'
    if review == Label.NONE:
        title = title.format('None')
    else:
        title = title.format(review.label_name.capitalize())

    windows['__WORDS'] = Win(None, title=title, rows=27, cols=win_width, y=9,
                             x=win_width, show_title=True)
    windows['__STATS'] = Win(None, rows=9, cols=win_width, y=0, x=win_width)
    return windows


def curses_main(scr, terms, args, review, last_reviews, logger=None,
                profiler=None):
    """
    Main loop

    :param scr: main window (the entire screen). It is passed by curses
    :type scr: _curses.window
    :param terms: list of terms
    :type terms: TermList
    :param args: command line arguments
    :type args: argparse.Namespace
    :param review: label to review if any
    :type review: Label
    :param last_reviews: last reviews performed. key: abs path of the csv; value: reviewed label name
    :type last_reviews: dict[str, str]
    :param logger: debug logger. Default: None
    :type logger: logging.Logger or None
    :param profiler: profiler logger. Default None
    :type profiler: logging.Logger or None
    """
    datafile = args.datafile
    confirmed = []
    reset = False
    if review != Label.NONE:
        # review mode: check last_reviews
        if review.label_name != last_reviews.get(datafile, ''):
            reset = True

        if reset:
            for w in terms.items:
                w.order = -1
                w.related = ''

    stdscr = init_curses()
    win_width = 40
    rows = 8

    # define windows
    windows = create_windows(win_width, rows, review)

    curses.ungetch(' ')
    _ = stdscr.getch()
    for win in windows:
        if win in ['__WORDS', '__STATS']:
            continue

        if win == review.label_name:
            # in review mode we must add to the window associated with the label
            # review only the items in confirmed (if any)
            conf_word = [w for w in terms.items if w.label == review and w.order >= 0]
            windows[win].assign_lines(conf_word)
        else:
            windows[win].assign_lines(terms.items)

    last_word = terms.get_last_classified_term()

    if last_word is None:
        refresh_label_windows('', Label.NONE, windows)
        related_items_count = 0
        sort_word_key = ''
        if review != Label.NONE:
            # review mode
            lines = []
            for w in terms.items:
                if w.label == review and w.order < 0:
                    lines.append(w.term)
        else:
            lines = [w.term for w in terms.items if not w.is_classified()]
    else:
        refresh_label_windows(last_word.term, last_word.label, windows)
        sort_word_key = last_word.related
        if sort_word_key == '':
            sort_word_key = last_word.term

        containing, not_containing = terms.return_related_items(sort_word_key,
                                                                label=review)
        related_items_count = len(containing)
        lines = containing
        lines.extend(not_containing)

    windows['__WORDS'].lines = lines
    windows['__STATS'].lines = get_stats_strings(terms, related_items_count)
    windows['__STATS'].display_lines(rev=False)
    classifing_keys = [Label.KEYWORD.key,
                       Label.NOT_RELEVANT.key,
                       Label.NOISE.key,
                       Label.RELEVANT.key]
    while True:
        if len(windows['__WORDS'].lines) <= 0:
            evaluated_word = ''
        else:
            evaluated_word = windows['__WORDS'].lines[0]

        if related_items_count <= 0:
            sort_word_key = ''

        windows['__WORDS'].display_lines(rev=False,
                                         highlight_word=sort_word_key)
        c = chr(stdscr.getch())
        if c not in ['w', 'q', 'u'] and evaluated_word == '':
            # no terms to classify. the only working keys are write, undo and
            # quit the others will do nothing
            continue

        if c in classifing_keys:
            label = Label.get_from_key(c)
            profiler.info("WORD '{}' AS '{}'".format(evaluated_word,
                                                     label.label_name))
            related_items_count, sort_word_key = do_classify(label, terms,
                                                             review,
                                                             evaluated_word,
                                                             sort_word_key,
                                                             related_items_count,
                                                             windows)
        elif c == 'p':
            profiler.info("WORD '{}' POSTPONED".format(evaluated_word))
            # classification: POSTPONED
            terms.classify_term(evaluated_word, Label.POSTPONED,
                                terms.get_last_classified_order() + 1,
                                sort_word_key)
            windows['__WORDS'].lines = windows['__WORDS'].lines[1:]
            windows[Label.POSTPONED.label_name].lines.append(evaluated_word)
            refresh_label_windows(evaluated_word, Label.POSTPONED, windows)
            related_items_count -= 1
        elif c == 'w':
            # write to file
            terms.to_csv(datafile)
        elif c == 'u':
            # undo last operation
            related_items_count, sort_word_key = undo(terms, review,
                                                      sort_word_key,
                                                      related_items_count,
                                                      windows, logger, profiler)

        elif c == 'q':
            # quit
            break
        else:
            # no recognized key: doing nothing (and avoiding useless autosave)
            continue

        windows['__STATS'].lines = get_stats_strings(terms, related_items_count)
        windows['__STATS'].display_lines(rev=False)

        if not args.dry_run and not args.no_auto_save:
            terms.to_csv(datafile)


def main():
    """
    Main function
    """
    parser = init_argparser()
    args = parser.parse_args()

    if args.no_profile:
        profile_log_level = logging.CRITICAL
    else:
        profile_log_level = logging.INFO

    profiler_logger = setup_logger('profiler_logger', 'profiler.log',
                                   level=profile_log_level)
    debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                                level=logging.DEBUG)

    if args.input is not None:
        try:
            review = Label.get_from_name(args.input)
        except ValueError:
            debug_logger.error('{} is not a valid label'.format(args.input))
            sys.exit('Error: {} is not a valid label'.format(args.input))
    else:
        review = Label.NONE

    profiler_logger.info("*** PROGRAM STARTED ***")
    datafile_path = str(pathlib.Path(args.datafile).absolute())
    profiler_logger.info("DATAFILE: '{}'".format(datafile_path))
    # use the absolute path
    args.datafile = datafile_path
    terms = TermList()
    _, _ = terms.from_csv(args.datafile)
    profiler_logger.info("CLASSIFIED: {}".format(terms.count_classified()))
    # check the last_review file
    try:
        with open('last_review.json') as file:
            last_reviews = json.load(file)
    except FileNotFoundError:
        # no file to care about
        last_reviews = dict()

    if review != Label.NONE:
        label = review.label_name
    else:
        label = 'NONE'
        if datafile_path in last_reviews:
            # remove the last review on the same csv
            del last_reviews[datafile_path]
            if len(last_reviews) <= 0:
                try:
                    os.unlink('last_review.json')
                except FileNotFoundError:
                    pass
            # also reset order and related
            for t in terms.items:
                t.order = -1
                t.related = ''

    profiler_logger.info("INPUT LABEL: {}".format(label))

    curses.wrapper(curses_main, terms, args, review, last_reviews,
                   logger=debug_logger, profiler=profiler_logger)

    profiler_logger.info("CLASSIFIED: {}".format(terms.count_classified()))
    profiler_logger.info("DATAFILE '{}'".format(datafile_path))
    profiler_logger.info("*** PROGRAM TERMINATED ***")
    curses.endwin()

    if review != Label.NONE:
        # ending review mode we must save some info
        last_reviews[datafile_path] = review.label_name

    if len(last_reviews) > 0:
        with open('last_review.json', 'w') as fout:
            json.dump(last_reviews, fout)

    if not args.dry_run:
        terms.to_csv(args.datafile)


if __name__ == "__main__":
    main()
