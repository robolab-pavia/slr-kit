import sys
import csv
import curses


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
    return (header, items)


def write_words(outfile):
    with open(outfile, mode='w') as out:
        employee_writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(header)
        for w in words:
            employee_writer.writerow(w)


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

    def display_words(self):
        for i, w in enumerate(reversed(self.words)):
            self.win_handler.addstr(i + 1, 1, w + ' '*(self.cols - 2 - len(w)))
            if i >= self.rows - 3:
                break
        self.win_handler.border()
        self.win_handler.refresh()

    def assign_words(self, words):
        self.words = [w[0] for w in words if w[2] == self.key]
        #print(self.words)


def get_next_word_without_group(words, i):
    j = i
    for w in words[i:]:
        if w[2] == '':
            return w[0], j
        j += 1


def avg_or_zero(num, den):
    if den > 0:
        avg = 100 * num / den
    else:
        avg = 0
    return avg


def update_stats(stats_window, words):
    string = 'Total words:  {:7}'.format(len(words))
    stats_window.win_handler.addstr(1, 1, string + ' '*(stats_window.cols - 2 - len(string)))
    n_completed = len([w for w in words if w[2] != ''])
    avg = avg_or_zero(n_completed, len(words))
    string = 'Completed:    {:7} ({:5.2f}%)'.format(n_completed, avg)
    stats_window.win_handler.addstr(2, 1, string + ' '*(stats_window.cols - 2 - len(string)))
    n_keywords = len([w for w in words if w[2] == 'k'])
    avg = avg_or_zero(n_keywords, n_completed)
    string = 'Keywords:     {:7} ({:5.2f}%)'.format(n_keywords, avg)
    stats_window.win_handler.addstr(3, 1, string + ' '*(stats_window.cols - 2 - len(string)))
    n_noise = len([w for w in words if w[2] == 'n'])
    avg = avg_or_zero(n_noise, n_completed)
    string = 'Noise:        {:7} ({:5.2f}%)'.format(n_noise, avg)
    stats_window.win_handler.addstr(4, 1, string + ' '*(stats_window.cols - 2 - len(string)))
    n_not_relevant = len([w for w in words if w[2] == 'x'])
    avg = avg_or_zero(n_not_relevant, n_completed)
    string = 'Not relevant: {:7} ({:5.2f}%)'.format(n_not_relevant, avg)
    stats_window.win_handler.addstr(5, 1, string + ' '*(stats_window.cols - 2 - len(string)))
    stats_window.win_handler.border()
    stats_window.win_handler.refresh()


def main(args, words):
    # create stdscr
    stdscr = curses.initscr()
    stdscr.clear()

    # allow echo, set colors
    curses.noecho()
    curses.start_color()
    curses.use_default_colors()

    win_width = 40

    # define windows
    windows = {
        'k': Win('k', title='Keywords', rows=12, cols=win_width, y=3, x=0),
        'n': Win('n', title='Noise', rows=12, cols=win_width, y=15, x=0),
        'x': Win('x', title='Not-relevant', rows=12, cols=win_width, y=27, x=0)
    }
    curses.ungetch(' ')
    c = stdscr.getch()
    for win in windows:
        windows[win].assign_words(words)
        windows[win].display_words()
    command_window = Win(None, rows=3, cols=2*win_width, y=0, x=0)
    stats_window = Win(None, rows=7, cols=win_width, y=3, x=win_width)
    update_stats(stats_window, words)

    i = 0
    while 1:
        word, i = get_next_word_without_group(words, i)
        command_window.win_handler.addstr(1, 1, word + ' '*(command_window.cols - 2 - len(word)))
        command_window.win_handler.border()
        command_window.win_handler.refresh()
        c = stdscr.getch()
        if c in [ord('k'), ord('n'), ord('x')]:
            words[i][2] = chr(c)
            win = windows[chr(c)]
            win.words.append(word)
            win.display_words()
            i += 1
        elif c == ord(' '):
            i += 1
        elif c == ord('q'):
            break
        update_stats(stats_window, words)

    curses.endwin()

(header, words) = load_words(sys.argv[1])

curses.wrapper(main, words)

write_words(sys.argv[1])
