import argparse
import curses
import logging

# List of class names
from words import Label, WordList


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
    n_keywords = words.count_by_class(Label.KEYWORD)
    n_noise = words.count_by_class(Label.NOISE)
    n_not_relevant = words.count_by_class(Label.NOT_RELEVANT)
    n_later = words.count_by_class(Label.POSTPONED)
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
    windows[klass.name].lines.append(evaluated_word)
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
        if win == klass.name:
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
    words.mark_word(last_word.word, Label.NONE, None)
    # remove last_word from the window that actually contains it
    try:
        win = windows[group.name]
        win.lines.remove(last_word.word)
        prev_last_word = words.get_last_inserted_word()
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
        containing, not_containing = words.return_related_items(sort_word_key)
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


def create_windows(win_width, rows):
    windows = dict()
    win_classes = [Label.KEYWORD, Label.RELEVANT, Label.NOISE,
                   Label.NOT_RELEVANT, Label.POSTPONED]
    for i, cls in enumerate(win_classes):
        windows[cls.name] = Win(cls, title=cls.name.capitalize(),
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

    last_word = words.get_last_inserted_word()
    if last_word is None:
        refresh_class_windows('', Label.NONE, windows)
        related_items_count = 0
        sort_word_key = ''
        lines = [w.word for w in words.items if not w.is_grouped()]
    else:
        refresh_class_windows(last_word.word, last_word.group, windows)
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
        classifing_keys = [Label.KEYWORD.key,
                           Label.NOT_RELEVANT.key,
                           Label.NOISE.key,
                           Label.RELEVANT.key]
        if c in classifing_keys:
            klass = Label.get_from_key(c)
            profiler.info("WORD '{}' AS '{}'".format(evaluated_word,
                                                     klass.name))
            related_items_count, sort_word_key = do_classify(klass, words,
                                                             evaluated_word,
                                                             sort_word_key,
                                                             related_items_count,
                                                             windows)
        elif c == 'p':
            profiler.info("WORD '{}' POSTPONED".format(evaluated_word))
            # classification: POSTPONED
            words.mark_word(evaluated_word, Label.POSTPONED,
                            words.get_last_inserted_order() + 1,
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
