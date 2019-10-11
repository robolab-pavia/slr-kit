import argparse
import curses
import json
import logging
import os
import sys
from words import Label, WordList


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
        :type label: Label
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
        self.group = label
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

    def display_lines(self, rev=True, highlight_word='', color_pair=1):
        """
        Display the lines associated to the window

        :param rev: if True, display lines in reversed order. Default: True
        :type rev: bool
        :param highlight_word: the word to highlight. Default: empty string
        :type highlight_word: str
        :param color_pair: the curses color pair to use to hightlight. Default 1
        :type color_pair: int
        """
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
                if w == highlight_word:
                    self.win_handler.addstr(i + 1, 1,
                                            trunc_w + ' ' * (self.cols - 2 - len(trunc_w)),
                                            curses.color_pair(color_pair))
                else:
                    tok = w.split(highlight_word)
                    if len(tok) == 1:
                        # no highlight_word found
                        self.win_handler.addstr(i + 1, 1,
                                                trunc_w + ' ' * (self.cols - 2 - len(trunc_w)))
                    else:
                        self.win_handler.addstr(i + 1, 1, '')
                        line = ''
                        for t in tok:
                            line += t
                            line += highlight_word
                            self.win_handler.addstr(t)
                            self.win_handler.addstr(highlight_word,
                                                    curses.color_pair(color_pair))

                        self.win_handler.addstr(i + 1, len(trunc_w) + 1,
                                                ' ' * (self.cols - 2 - len(trunc_w)))
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

    def assign_lines(self, terms):
        """
        Assign the terms in terms with the same label as the window
        :param terms: the terms list
        :type terms: list[Word]
        """
        self.lines = [w.word for w in terms if w.group == self.group]


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


def get_stats_strings(words, related_items_count=0):
    """
    Calculates the statistics and formats them into strings

    :param words: the list of terms
    :type words: WordList
    :param related_items_count: the current number of related term
    :type related_items_count: int
    :return: the statistics about words formatted as strings
    :rtype: list[str]
    """
    stats_strings = []
    n_completed = words.count_classified()
    n_keywords = words.count_by_label(Label.KEYWORD)
    n_noise = words.count_by_label(Label.NOISE)
    n_not_relevant = words.count_by_label(Label.NOT_RELEVANT)
    n_later = words.count_by_label(Label.POSTPONED)
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


def do_classify(klass, words, review, evaluated_term, sort_word_key,
                related_items_count, windows):
    """
    Handle the term classification process of the evaluated_term

    :param klass: label to be assigned to the evaluated_term
    :type klass: Label
    :param words: the list of terms
    :type words: WordList
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
    windows[klass.name].lines.append(evaluated_term)
    refresh_class_windows(evaluated_term, klass, windows)

    words.classify_term(evaluated_term, klass,
                        words.get_last_classified_order() + 1, sort_word_key)

    if related_items_count <= 0:
        sort_word_key = evaluated_term

    containing, not_containing = words.return_related_items(sort_word_key,
                                                            label=review)

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
        if win == klass.name:
            windows[win].display_lines(rev=True, highlight_word=evaluated_word,
                                       color_pair=2)
        else:
            windows[win].display_lines(rev=True)


def undo(words, review, sort_word_key, related_items_count, windows, logger,
         profiler):
    """
    Handle the undo of a term

    :param words: the list of terms
    :type words: WordList
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
    last_word = words.get_last_classified_word()
    if last_word is None or last_word.group == review:
        return related_items_count, sort_word_key

    group = last_word.group
    related = last_word.related
    logger.debug("Undo: {} group {} order {}".format(last_word.word,
                                                     group,
                                                     last_word.order))
    # un-mark last_word
    words.classify_term(last_word.word, review, -1)
    # remove last_word from the window that actually contains it
    try:
        win = windows[group.name]
        win.lines.remove(last_word.word)
        prev_last_word = words.get_last_classified_word()
        if prev_last_word is not None:
            refresh_class_windows(prev_last_word.word, prev_last_word.group,
                                  windows)
        else:
            refresh_class_windows('', Label.NONE, windows)
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
        containing, not_containing = words.return_related_items(sort_word_key,
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

    profiler.info("WORD '{}' UNDONE".format(last_word.word))

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
        windows[cls.name] = Win(cls, title=cls.name.capitalize(),
                                rows=rows, cols=win_width, y=(rows + 1) * i,
                                x=0, show_title=True)

    title = 'Input label: {}'
    if review == Label.NONE:
        title = title.format('None')
    else:
        title = title.format(review.classname.capitalize())

    windows['__WORDS'] = Win(None, title=title, rows=27, cols=win_width, y=9,
                             x=win_width, show_title=True)
    windows['__STATS'] = Win(None, rows=9, cols=win_width, y=0, x=win_width)
    return windows


def curses_main(scr, words, args, review, logger=None, profiler=None):
    """
    Main loop

    :param scr: main window (the entire screen). It is passed by curses
    :type scr: _curses.window
    :param words: list of terms
    :type words: WordList
    :param args: command line arguments
    :type args: argparse.Namespace
    :param review: label to review if any
    :type review: Label
    :param logger: debug logger. Default: None
    :type logger: logging.Logger or None
    :param profiler: profiler logger. Default None
    :type profiler: logging.Logger or None
    """
    datafile = args.datafile
    confirmed = []
    if review != Label.NONE:
        # review mode: retrieve some info and reset order and related
        try:
            with open('last_review.json') as fin:
                data = json.load(fin)
                if review.classname == data['label']:
                    confirmed = data['confirmed']
                # else: last review was about another label so confirmed must be
                # empty: nothing to do
        except FileNotFoundError:
            # no last review so confirmed must be empty: nothing to do
            pass

        # FIXME: method in WordList
        for w in words.items:
            if w.word in confirmed:
                w.order = 0
            else:
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

        if win == review.classname:
            # in review mode we must add to the window associated with the label
            # review only the items in confirmed (if any)
            conf_word = [w for w in words.items if w.word in confirmed]
            windows[win].assign_lines(conf_word)
        else:
            windows[win].assign_lines(words.items)

    if review == Label.NONE:
        last_word = words.get_last_classified_word()
    else:
        last_word = None

    if last_word is None:
        refresh_class_windows('', Label.NONE, windows)
        related_items_count = 0
        sort_word_key = ''
        if review != Label.NONE:
            # review mode
            # FIXME: better way?
            lines = []
            for w in words.items:
                if w.group == review and w.word not in confirmed:
                    lines.append(w.word)
        else:
            lines = [w.word for w in words.items if not w.is_classified()]
    else:
        refresh_class_windows(last_word.word, last_word.group, windows)
        sort_word_key = last_word.related
        if sort_word_key == '':
            sort_word_key = last_word.word

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
        classifing_keys = [Label.KEYWORD.key,
                           Label.NOT_RELEVANT.key,
                           Label.NOISE.key,
                           Label.RELEVANT.key]
        if c in classifing_keys:
            klass = Label.get_from_key(c)
            profiler.info("WORD '{}' AS '{}'".format(evaluated_word,
                                                     klass.name))
            related_items_count, sort_word_key = do_classify(klass, words,
                                                             review,
                                                             evaluated_word,
                                                             sort_word_key,
                                                             related_items_count,
                                                             windows)
        elif c == 'p':
            profiler.info("WORD '{}' POSTPONED".format(evaluated_word))
            # classification: POSTPONED
            words.classify_term(evaluated_word, Label.POSTPONED,
                                words.get_last_classified_order() + 1,
                                sort_word_key)
            windows['__WORDS'].lines = windows['__WORDS'].lines[1:]
            windows[Label.POSTPONED.name].lines.append(evaluated_word)
            refresh_class_windows(evaluated_word, Label.POSTPONED, windows)
            related_items_count -= 1
        elif c == 'w':
            # write to file
            words.to_csv(datafile)
        elif c == 'u':
            # undo last operation
            related_items_count, sort_word_key = undo(words, review,
                                                      sort_word_key,
                                                      related_items_count,
                                                      windows, logger, profiler)

        elif c == 'q':
            # quit
            break

        windows['__STATS'].lines = get_stats_strings(words, related_items_count)
        windows['__STATS'].display_lines(rev=False)

        if not args.dry_run and not args.no_auto_save:
            words.to_csv(datafile)


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
            review = Label.get_from_classname(args.input)
        except ValueError:
            debug_logger.error('{} is not a valid label'.format(args.input))
            sys.exit('Error: {} is not a valid label'.format(args.input))
    else:
        review = Label.NONE

    profiler_logger.info("*** PROGRAM STARTED ***".format(args.datafile))
    profiler_logger.info("DATAFILE: '{}'".format(args.datafile))
    words = WordList()
    _, _ = words.from_csv(args.datafile)
    profiler_logger.info("CLASSIFIED: {}".format(words.count_classified()))
    if review != Label.NONE:
        label = review.classname
    else:
        label = 'NONE'

    profiler_logger.info("INPUT LABEL: {}".format(label))

    curses.wrapper(curses_main, words, args, review,
                   logger=debug_logger, profiler=profiler_logger)

    profiler_logger.info("CLASSIFIED: {}".format(words.count_classified()))
    profiler_logger.info("DATAFILE '{}'".format(args.datafile))
    profiler_logger.info("*** PROGRAM TERMINATED ***")
    curses.endwin()

    if review != Label.NONE:
        # ending review mode we must save some info
        confirmed = []
        for w in words.items:
            if w.group == review and w.order is not None:
                confirmed.append(w.word)

        data = {'label': review.classname,
                'confirmed': confirmed}

        with open('last_review.json', 'w') as fout:
            json.dump(data, fout)
    else:
        if os.path.exists('last_review.json'):
            # if no review we must delete the previous last_review.json to avoid
            # problems in future reviews
            os.unlink('last_review.json')

    if not args.dry_run:
        words.to_csv(args.datafile)


if __name__ == "__main__":
    main()
