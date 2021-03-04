import argparse
import json
import logging
import os
import pathlib
import sys
from typing import cast, Callable, Hashable

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Dimension, Window, Layout
from prompt_toolkit.layout.containers import Container, VSplit, HSplit
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.widgets import TextArea, Frame

from terms import Label, TermList, Term
from utils import setup_logger, substring_index
import time

debug_logger = setup_logger('debug_logger', 'slr-kit.log',
                            level=logging.DEBUG)

DEBUG = False


class TermLexer(Lexer):
    def invalidation_hash(self) -> Hashable:
        return self._inv

    def __init__(self):
        self._word = ''
        self._color = 'ffffff'
        self._inv = 0
        self._whole_line = False
        self._show_header = False
        self._highlight_first = False

    @property
    def highlight_first(self) -> bool:
        return self._highlight_first

    @highlight_first.setter
    def highlight_first(self, highlight_first: bool):
        self._highlight_first = highlight_first
        self._handle_inv()

    @property
    def show_header(self) -> bool:
        return self._show_header

    @show_header.setter
    def show_header(self, show_header: bool):
        self._show_header = show_header
        self._handle_inv()

    @property
    def word(self) -> str:
        return self._word

    @word.setter
    def word(self, word: str):
        self._word = word
        self._handle_inv()

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str):
        self._color = color
        self._handle_inv()

    def _handle_inv(self):
        self._inv += 1
        # avoid overflow problems
        if self._inv > 10 * 1000 * 1000:
            self._inv = 0

    @property
    def whole_line(self) -> bool:
        """
        If the lexer must highlight only the whole line == to word

        If True only a line equal to word is highlighted. If False the lexer
        will highlight word in all the lines that contains it
        :return: if the lexer must highlight only the whole line == to word
        :rtype: bool
        """
        return self._whole_line

    @whole_line.setter
    def whole_line(self, whole: bool):
        self._whole_line = whole
        self._inv += 1

    def lex_document(self, document):
        lines = []
        for i, line in enumerate(document.lines):
            fmt = []
            if i == 0 and self.show_header:
                lines.append([('underline', line)])
                continue

            fmt_first = ''
            if i == 1 and self._highlight_first:
                fmt_first = 'underline'

            prev = 0
            if self.whole_line:
                if line == self.word:
                    fmt.append((f'#{self.color} bold', line))
                else:
                    fmt.append(('', line))
            else:
                for begin, end in substring_index(line, self.word):
                    if begin > prev:
                        fmt.append((f'{fmt_first}', line[prev:begin]))

                    fmt.append((f'#{self.color} bold {fmt_first}',
                                line[begin:end]))
                    prev = end

                if prev < len(line) - 1:
                    fmt.append((f'{fmt_first}', line[prev:]))

            lines.append(fmt)

        return lambda lineno: lines[lineno]


class Win:
    """
    Window that shows terms

    :type label: Label or None
    :type title: str
    :type show_title: bool
    :type lexer: TermLexer
    :type height: Dimension
    :type width: Dimension
    :type buffer: Buffer
    :type control: BufferControl
    :type window: Window
    :type terms: list[Term] or None
    :type attr: FormattedTextControl or None
    """

    def __init__(self, label, title='', rows=3, cols=30, show_title=False,
                 show_count=False, show_label=False, show_header=False,
                 highlight_first=False):
        """
        Creates a window that shows terms

        :param label: label associated to the windows
        :type label: Label or None
        :param title: title of window. Default: empty string
        :type title: str
        :param rows: number of rows
        :type rows: int
        :param cols: number of columns
        :type cols: int
        :param show_title: if True the window shows its title. Default: False
        :type show_title: bool
        :param show_header: if the header must be shown
        :type show_header: bool
        :param highlight_first: if true the first word is highlighted
        :type highlight_first: bool
        """
        self.highlight_first = highlight_first
        if show_count and show_label:
            raise ValueError('show_count and show_label cannot be both set')

        self.label = label
        self.title = title
        self.show_title = show_title
        self.lexer = TermLexer()

        # we must re-create a text-area using the basic components
        # we must do this to have control on the lexer. Otherwise prompt-toolkit
        # will cache the output of the lexer resulting in wrong highlighting
        self.buffer = Buffer(read_only=True, document=Document('', 0))
        self.control = BufferControl(buffer=self.buffer,
                                     lexer=self.lexer)

        if show_count:
            attr_width = 6
            self.attr_name = 'count'
        elif show_label:
            attr_width = 10
            self.attr_name = 'label'
        else:
            attr_width = 0
            self.attr_name = ''

        cols -= attr_width
        if cols <= 0:
            raise ValueError('cols is too small')

        self.height = Dimension(preferred=rows, max=rows)
        self.width = Dimension(preferred=cols, max=cols)
        self.window = Window(content=self.control, height=self.height,
                             width=self.width)
        split = [
            cast('Container', self.window),
        ]
        if show_count or show_label:
            self.attr = FormattedTextControl()
            split.append(cast('Container', Window(self.attr,
                                                  width=attr_width)))
        else:
            self.attr = None

        self.show_header = show_header
        self.terms = None
        self.frame = Frame(cast('Container', VSplit(split)))
        if self.show_title:
            self.frame.title = title

    def __pt_container__(self) -> Container:
        return self.frame.__pt_container__()

    @property
    def text(self) -> str:
        """
        Text shown in the window

        :return: the text shown in the window
        """
        return self.buffer.text

    @text.setter
    def text(self, value: str):
        """
        Sets the text to be shown

        :param value: the new text to be shown
        :type value: str
        """
        self.buffer.set_document(Document(value, 0), bypass_readonly=True)

    def assign_terms(self, terms, classified=False):
        """
        Assigns the terms to the window

        :param terms: terms to assign
        :type terms: TermList
        :param classified: if True, only the classified terms are assigned
        :type classified: bool
        """
        if classified:
            self.terms = terms.get_classified().items
        else:
            self.terms = terms.get_not_classified().items

    def assign_lines(self, terms):
        """
        Assign the terms in terms with the same label as the window

        :param terms: the terms list
        :type terms: list[Term]
        """
        self.terms = [w for w in terms if w.label == self.label]
        self.terms = sorted(self.terms, key=lambda t: t.order)

    def display_lines(self, rev=True, highlight_word='', whole_line=False,
                      color='ff0000'):
        """
        Display the terms associated to the window

        :param rev: if True, display terms in reversed order. Default: True
        :type rev: bool
        :param highlight_word: the word to highlight. Default: empty string
        :type highlight_word: str
        :param whole_line: if True only a line == highlight_word is highlighted.
        :type whole_line: bool
        :param color: hex code for the highlight color. Default red (ff0000)
        :type color: str
        """
        terms = iter(self.terms)
        if rev:
            terms = reversed(self.terms)

        self.lexer.word = highlight_word
        self.lexer.color = color
        self.lexer.whole_line = whole_line
        self.lexer.show_header = self.show_header
        self.lexer.highlight_first = self.highlight_first
        if self.show_header:
            text = ['Term']
            attr = [('underline', f'{self.attr_name.title()}\n')]
        else:
            text = []
            attr = []

        for w in terms:
            text.append(w.string)
            if self.attr_name == 'count':
                attr.append(('', f'{w.count}\n'))
            else:
                attr.append(('', f'{w.label.label_name}\n'))

            # noinspection PyTypeChecker
            if len(text) >= self.height.max:
                break

        self.text = '\n'.join(text)
        if self.attr is not None:
            self.attr.text = attr


class StrWin:
    """
    Window that shows strings

    :type height: Dimension
    :type width: Dimension
    """

    def __init__(self, rows=3, cols=30):
        """
        Creates a window that shows strings

        :param rows: number of rows
        :type rows: int
        :param cols: number of columns
        :type cols: int
        """
        self.height = Dimension(preferred=rows, max=rows)
        self.width = Dimension(preferred=cols, max=cols)
        self.textarea = TextArea(height=self.height, width=self.width,
                                 read_only=True)
        self.strings = None
        self.frame = Frame(cast('Container', self.textarea))

    def __pt_container__(self) -> Container:
        return self.frame.__pt_container__()

    def assign_lines(self, lines):
        """
        Assign the lines to the window

        :param lines: the lines to show
        :type lines: list[str]
        """
        self.strings = lines

    @property
    def text(self):
        """
        Gets the text shown in the windows

        :return: the text shown in the windows
        :rtype: str
        """
        return self.textarea.text

    @text.setter
    def text(self, text):
        """
        Sets the text to be shown

        :param text: the text to be shown
        :type text: str
        """
        self.textarea.text = text


class Gui:
    """
    :type _word_win: Win or None
    :type _stats_win: StrWin or None
    :type _class_win: Win or None
    :type _post_win: Win or None
    """

    def __init__(self, width, term_rows, rows, review):
        """
        The Gui of the application

        :param width: width of each windows
        :type width: int
        :param term_rows: number of row of the term window
        :type term_rows: int
        :param rows: number of row of all the other windows
        :type rows: int
        :param review: label to review
        :type review: Label
        """
        self._class_win = None
        self._post_win = None
        self._word_win = None
        self._stats_win = None
        self._create_windows(width, term_rows, rows, review)
        self._review = review
        self._body = VSplit(
            [
                HSplit([
                    cast('Container', self._class_win),
                    cast('Container', self._post_win),
                ]),
                HSplit([
                    cast('Container', self._stats_win),
                    cast('Container', self._word_win)
                ])
            ]
        )

    @property
    def body(self):
        return self._body

    def _create_windows(self, win_width, term_rows, rows, review):
        """
        Creates all the windows

        :param win_width: number of columns of each windows
        :type win_width: int
        :param term_rows: number of row of the term window
        :type term_rows: int
        :param rows: number of row of each windows
        :type rows: int
        :param review: label to review
        :type review: Label
        """
        self._class_win = Win(Label.NONE, title='Classified terms',
                              rows=term_rows, cols=win_width,
                              show_title=True, show_label=True,
                              show_header=True)
        title = Label.POSTPONED.label_name.capitalize()
        self._post_win = Win(Label.POSTPONED, title=title, rows=rows,
                             cols=win_width,
                             show_title=True)

        title = 'Input label: {}'
        if review == Label.NONE:
            title = title.format('None')
        else:
            title = title.format(review.label_name.capitalize())

        self._word_win = Win(Label.NONE, title=title, rows=term_rows,
                             cols=win_width, show_title=True, show_count=True,
                             show_header=True, highlight_first=True)
        self._stats_win = StrWin(rows=rows, cols=win_width)

    def refresh_label_windows(self, term_to_highlight, label):
        """
        Refresh the windows associated with a label

        :param term_to_highlight: the term to highlight
        :type term_to_highlight: str
        :param label: label of the window that has to highlight the term
        :type label: Label
        """
        if label == Label.POSTPONED:
            self._post_win.display_lines(rev=True,
                                         highlight_word=term_to_highlight,
                                         whole_line=True, color='ffff00')
            self._class_win.display_lines(rev=True)
        else:
            self._class_win.display_lines(rev=True,
                                          highlight_word=term_to_highlight,
                                          whole_line=True,
                                          color='ffff00')
            self._post_win.display_lines(rev=True)

    def set_stats(self, stats):
        """
        Sets the statistics in the proper windows

        :param stats: the statistics about words formatted as strings
        :type stats: list[str]
        """
        self._stats_win.text = '\n'.join(stats)

    def set_terms(self, to_classify, sort_key):
        """
        Sets the terms to be classified in the proper window

        :param to_classify: list of terms to be classified
        :type to_classify: TermList
        :param sort_key: string used for searching the related terms
        :type sort_key: str
        """
        self._word_win.terms = to_classify.items
        self._word_win.display_lines(rev=False, highlight_word=sort_key)

    def update_windows(self, to_classify, classified, postponed,
                       term_to_highlight, sort_word_key, stats_str):
        """
        Handle the update of all the windows

        :param to_classify: terms not yet classified
        :type to_classify: TermList
        :param classified: list of classified terms
        :type classified: TermList
        :param postponed: list of postponed terms
        :type postponed: TermList
        :param term_to_highlight: term to hightlight as the last classified term
        :type term_to_highlight: Term
        :param sort_word_key: words used for the related item highlighting
        :type sort_word_key: str
        :param stats_str: statistics to display
        :type stats_str: list[str]
        """
        self.set_terms(to_classify, sort_word_key)

        self._post_win.terms = postponed.items

        self._class_win.terms = classified.items

        if term_to_highlight is not None:
            self.refresh_label_windows(term_to_highlight.string,
                                       term_to_highlight.label)
        else:
            self.refresh_label_windows('', Label.NONE)

        self.set_stats(stats_str)

    def assign_labeled_terms(self, classified, postponed):
        """
        Assigns the labeled terms to the correct windows

        :param classified: the list of classified term
        :type classified: TermList
        :param postponed: the list of postponed term
        :type postponed: TermList
        """
        self._class_win.assign_terms(classified, classified=True)
        self._post_win.assign_terms(postponed, classified=True)


class Fawoc:
    def __init__(self, args, terms, review, gui, profiler, logger):
        """
        The fawoc application logic

        :param args: command line arguments
        :type args: argparse.Namespace
        :param terms: list of terms
        :type terms: TermList
        :param review: label to review
        :type review: Label
        :param gui: gui object of the application
        :type gui: Gui
        :param profiler: fawoc profiler
        :type profiler: logging.Logger
        :param logger: fawoc logger
        :type logger: logging.Logger
        """
        self.keybindings = KeyBindings()
        self.app = Application(layout=Layout(gui.body),
                               key_bindings=self.keybindings,
                               full_screen=True)
        self.args = args
        self.gui = gui
        self.terms = terms
        self.n_terms = len(terms)
        last_word = terms.get_last_classified_term()
        self.postponed = terms.get_from_label(Label.POSTPONED)
        if last_word is None:
            self.last_classified_order = -1
            related = 0
            sort_key = ''
            if review == Label.NONE:
                self.to_classify = terms.get_not_classified()
            else:
                self.to_classify = terms.get_from_label(review, order_set=False)
                self.to_classify.sort_by_index()
        else:
            self.last_classified_order = last_word.order
            sort_key = last_word.related
            if sort_key == '' and last_word.label != Label.POSTPONED:
                sort_key = last_word.string

            cont, not_cont = terms.return_related_items(sort_key, label=review)
            related = len(cont)
            self.to_classify = cont + not_cont

        cls = terms.get_classified()
        if review != Label.NONE:
            self.postponed = terms.get_from_label(Label.POSTPONED,
                                                  order_set=True)
            cls = TermList([t for t in cls.items if t.order >= 0])

        self.classified = cls - self.postponed

        self._get_next_word()
        self.review = review
        self.related_count = related
        if related > 0:
            self.sort_word_key = sort_key
        else:
            self.sort_word_key = ''

        self.last_word = last_word
        self.profiler = profiler
        self.logger = logger
        self.gui.assign_labeled_terms(self.classified, self.postponed)
        if self.last_word is None:
            self.gui.refresh_label_windows('', Label.NONE)
        else:
            self.gui.refresh_label_windows(self.last_word.string,
                                           self.last_word.label)

        self.gui.set_terms(self.to_classify, self.sort_word_key)
        self.gui.set_stats(self.get_stats_strings())

    def add_key_binding(self, keys, handler):
        """
        Adds keybinding to Fawoc.

        If any elements in keys is a single letter the function add the same
        handler to the letter and to the case-swapped letter

        :param keys: list of keys to be bound
        :type keys: list[str]
        :param handler: function to be call
        :type handler: Callable[[KeyPressEvent], None]
        """
        for k in keys:
            if len(k) == 1:
                if k.islower():
                    other = k.upper()
                else:
                    other = k.lower()

                self.keybindings.add(other)(handler)

            self.keybindings.add(k)(handler)

    def do_autonoise(self):
        """
        Classifies all the subsequent terms with the same num of word as autonoise
        """
        if self.evaluated_word is None:
            return

        n = len(self.evaluated_word.string.split())
        auto = []
        for t in self.to_classify.items:
            if len(t.string.split()) != n:
                break

            t.label = Label.AUTONOISE
            self.last_classified_order += 1
            t.related = self.sort_word_key
            t.order = self.last_classified_order
            auto.append(t.string)
            msg = f"WORD '{t.string}' labeled as {Label.AUTONOISE.label_name}"
            self.profiler.info(msg)
            self.classified.items.append(t)

        ret = self.terms.return_related_items(self.sort_word_key,
                                              label=self.review)
        containing, not_containing = ret
        self.related_count = len(containing)
        if self.related_count == 0:
            self.sort_word_key = ''

        self.to_classify = containing + not_containing
        self.last_word = self.classified.items[-1]
        stats = self.get_stats_strings()
        self.gui.update_windows(self.to_classify, self.classified,
                                self.postponed, self.last_word,
                                self.sort_word_key, stats)

        if not self.args.dry_run and not self.args.no_auto_save:
            self.save_terms()

        self._get_next_word()

    def do_classify(self, label):
        """
        Classify the evaluated word

        :param label: label to use to classify the term
        :type label: Label
        """
        debug_logger.debug("do_classify {} {}".format(self.evaluated_word, label))
        if self.evaluated_word is None:
            return

        t0 = time.time()
        self.profiler.info("WORD '{}' AS '{}'".format(self.evaluated_word.string,
                                                      label.label_name))

        self.evaluated_word.label = label
        self.last_classified_order += 1
        self.evaluated_word.order = self.last_classified_order
        self.evaluated_word.related = self.sort_word_key
        del self.to_classify.items[0]
        self.related_count -= 1
        t1 = time.time()
        debug_logger.debug("do_classify time 0 {}".format(t1 - t0))

        t0 = time.time()
        if self.related_count < 0:
            self.sort_word_key = self.evaluated_word.string
            ret = self.to_classify.return_related_items(self.sort_word_key,
                                                        label=self.review)
            containing, not_containing = ret
            self.to_classify = containing + not_containing
            # the sort_word_key has been changed: reload the related count
            self.related_count = len(containing)

        if self.related_count == 0:
            # last related word has been classified or no related words found:
            # reset the related machinery
            self.sort_word_key = ''

        t1 = time.time()
        debug_logger.debug("do_classify time 1 {}".format(t1 - t0))

        t0 = time.time()
        self.last_word = self.evaluated_word

        self.classified.items.append(self.evaluated_word)
        stats = self.get_stats_strings()
        self.gui.update_windows(self.to_classify, self.classified,
                                self.postponed, self.last_word,
                                self.sort_word_key, stats)
        t1 = time.time()
        debug_logger.debug("do_classify time 2 {}".format(t1 - t0))

        t0 = time.time()
        if not self.args.dry_run and not self.args.no_auto_save:
            self.save_terms()

        self._get_next_word()
        t1 = time.time()
        debug_logger.debug("do_classify time 3 {}".format(t1 - t0))

    def do_postpone(self):
        """
        Postpones the evaluated word
        """
        if self.evaluated_word is None:
            return

        msg = "WORD '{}' POSTPONED".format(self.evaluated_word.string)
        self.profiler.info(msg)
        # classification: POSTPONED
        self.evaluated_word.label = Label.POSTPONED
        self.last_classified_order += 1
        self.evaluated_word.order = self.last_classified_order
        self.evaluated_word.related = self.sort_word_key

        self.related_count -= 1
        if self.related_count > 0:
            cont, not_cont = self.terms.return_related_items(self.sort_word_key,
                                                             self.review)
            self.to_classify = cont + not_cont
        else:
            self.to_classify = self.terms.get_from_label(self.review)
            # reset related machinery
            self.related_count = 0
            self.sort_word_key = ''

        self.last_word = self.evaluated_word
        self.postponed.items.append(self.evaluated_word)
        stats = self.get_stats_strings()
        self.gui.update_windows(self.to_classify, self.classified,
                                self.postponed, self.last_word,
                                self.sort_word_key, stats)

        if not self.args.dry_run and not self.args.no_auto_save:
            self.save_terms()

        self._get_next_word()

    def _get_next_word(self):
        """
        Handle the evaluated word
        """
        if len(self.to_classify) <= 0:
            self.evaluated_word = None
        else:
            self.evaluated_word = self.to_classify.items[0]

    def undo(self):
        """
        Handles the undo process
        """
        self.last_word = self.terms.get_last_classified_term()
        if self.last_word is None:
            return

        label = self.last_word.label
        if label == Label.AUTONOISE:
            for t in reversed(list(self.classified.items)):
                if t.label == Label.AUTONOISE:
                    self._undo_single()
                else:
                    break
        else:
            self._undo_single()

        stats = self.get_stats_strings()
        self.gui.update_windows(self.to_classify, self.classified,
                                self.postponed, self.last_word,
                                self.sort_word_key, stats)

        if not self.args.dry_run and not self.args.no_auto_save:
            self.save_terms()

        self._get_next_word()

    def _undo_single(self):
        """
        Handle the undo of a single term
        """
        self.last_word = self.terms.get_last_classified_term()
        if self.last_word is None:
            return

        label = self.last_word.label
        related = self.last_word.related
        msg = 'Undo: {} group {} order {}'.format(self.last_word.string, label,
                                                  self.last_word.order)
        self.logger.debug(msg)

        if label == Label.POSTPONED:
            self.postponed.remove([self.last_word.string])
        else:
            self.classified.remove([self.last_word.string])

        # un-mark self.last_word
        self.last_classified_order = self.last_word.order
        self.last_word.label = self.review
        self.last_word.order = -1
        self.last_word.related = ''

        # handle related word
        if related == self.sort_word_key:
            self.related_count += 1
            self.to_classify.items.insert(0, self.last_word)
        elif self.sort_word_key == self.last_word.string:
            # the sort_word_key is the word undone: reset the related machinery
            self.sort_word_key = ''
            self.related_count = 0
            self.to_classify.sort_by_index()
            self.to_classify.items.insert(0, self.last_word)
        else:
            self.sort_word_key = related
            ret = self.terms.return_related_items(self.sort_word_key,
                                                  label=self.review)
            containing, not_containing = ret
            self.related_count = len(containing)
            self.to_classify = containing + not_containing

        if self.sort_word_key == '':
            # if self.sort_word_key is empty there's no related item: fix the
            # related_items_count to the correct value of 0
            self.related_count = 0

        self.profiler.info("WORD '{}' UNDONE".format(self.last_word.string))
        self.last_word = self.terms.get_last_classified_term()

    def save_terms(self):
        """
        Saves the terms to file
        """
        self.terms.to_tsv(self.args.datafile)
        jsonfile = f'{self.args.datafile}.json'
        self.terms.save_service_data(jsonfile)

    def get_stats_strings(self):
        """
        Calculates the statistics and formats them into strings

        :return: the statistics about words formatted as strings
        :rtype: list[str]
        """
        stats_strings = []
        n_later = len(self.postponed)
        n_keywords = self.classified.count_by_label(Label.KEYWORD)
        n_relevant = self.classified.count_by_label(Label.RELEVANT)
        n_noise = self.classified.count_by_label(Label.NOISE)
        n_not_relevant = self.classified.count_by_label(Label.NOT_RELEVANT)
        n_autonoise = self.classified.count_by_label(Label.AUTONOISE)
        n_completed = n_relevant + n_keywords + n_noise + n_not_relevant
        n_completed += n_later + n_autonoise
        stats_strings.append(f'Total words:  {self.n_terms:7}')

        avg = avg_or_zero(n_completed, self.n_terms)
        stats_strings.append(f'Completed:    {n_completed:7} ({avg:6.2f}%)')

        avg = avg_or_zero(n_keywords, n_completed)
        stats_strings.append(f'Keywords:     {n_keywords:7} ({avg:6.2f}%)')

        avg = avg_or_zero(n_relevant, n_completed)
        stats_strings.append(f'Relevant:     {n_relevant:7} ({avg:6.2f}%)')

        avg = avg_or_zero(n_noise, n_completed)
        stats_strings.append(f'Noise:        {n_noise:7} ({avg:6.2f}%)')

        avg = avg_or_zero(n_not_relevant, n_completed)
        stats_strings.append(f'Not relevant: {n_not_relevant:7} ({avg:6.2f}%)')

        avg = avg_or_zero(n_autonoise, n_completed)
        stats_strings.append(f'Autonoise:    {n_autonoise:7} ({avg:6.2f}%)')

        avg = avg_or_zero(n_later, n_completed)
        stats_strings.append(f'Postponed:    {n_later:7} ({avg:6.2f}%)')

        s = 'Related:      {:7}'
        if self.related_count >= 0:
            s = s.format(self.related_count)
        else:
            s = s.format(0)

        stats_strings.append(s)
        return stats_strings


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
    wmin = 40
    wmax = 100
    wdef = 50

    def check_width(arg):
        try:
            w = int(arg)
            if w < wmin or w > wmax:
                msg = f'width {w} is not in range [{wmin}, {wmax}]'
            else:
                return w
        except ValueError:
            msg = f'width {arg} is not a valid int'

        raise argparse.ArgumentTypeError(msg)

    help_msg = f"""width of the windows in columns.
Must be in range [{wmin}, {wmax}]. Default {wdef} columns"""
    parser.add_argument('--width', '-w', action='store', type=check_width,
                        help=help_msg, default=wdef)
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


def autonoise_kb(event: KeyPressEvent, fawoc: Fawoc):
    """
    Callback for the autonoise key

    :param event: prompt_toolkit event associate to the key pressed
    :type event: KeyPressEvent
    :param fawoc: fawoc object
    :type fawoc: Fawoc
    """
    fawoc.do_autonoise()


def classify_kb(event: KeyPressEvent, fawoc: Fawoc):
    """
    Callback for the classifing keys

    :param event: prompt_toolkit event associate to the key pressed
    :type event: KeyPressEvent
    :param fawoc: fawoc object
    :type fawoc: Fawoc
    """
    label = Label.get_from_key(event.data.lower())
    fawoc.do_classify(label)


def postpone_kb(event: KeyPressEvent, fawoc: Fawoc):
    """
    Callback for the postpone key

    :param event: prompt_toolkit event associate to the key pressed
    :type event: KeyPressEvent
    :param fawoc: fawoc object
    :type fawoc: Fawoc
    """
    fawoc.do_postpone()


def undo_kb(event: KeyPressEvent, fawoc: Fawoc):
    """
    Callback for the undo key

    :param event: prompt_toolkit event associate to the key pressed
    :type event: KeyPressEvent
    :param fawoc: fawoc object
    :type fawoc: Fawoc
    """
    fawoc.undo()


def save_kb(event: KeyPressEvent, fawoc: Fawoc):
    """
    Callback for the save key

    :param event: prompt_toolkit event associate to the key pressed
    :type event: KeyPressEvent
    :param fawoc: fawoc object
    :type fawoc: Fawoc
    """
    fawoc.save_terms()


def quit_kb(event: KeyPressEvent):
    """
    Callback for the quit key

    :param event: prompt_toolkit event associate to the key pressed
    :type event: KeyPressEvent
    """
    event.app.exit()


def fawoc_main(terms, args, review, last_reviews, logger=None, profiler=None):
    """
    Main loop

    :param terms: list of terms
    :type terms: TermList
    :param args: command line arguments
    :type args: argparse.Namespace
    :param review: label to review if any
    :type review: Label
    :param last_reviews: last reviews performed. key: csv abs path; val: reviewed label name
    :type last_reviews: dict[str, str]
    :param logger: debug logger. Default: None
    :type logger: logging.Logger or None
    :param profiler: profiler logger. Default None
    :type profiler: logging.Logger or None
    """
    datafile = args.datafile
    reset = False
    if review != Label.NONE:
        # review mode: check last_reviews
        if review.label_name != last_reviews.get(datafile, ''):
            reset = True

        if reset:
            for w in terms.items:
                w.order = -1
                w.related = ''

    win_width = args.width
    rows = 9
    terms_rows = 28

    gui = Gui(win_width, terms_rows, rows, review)
    fawoc = Fawoc(args, terms, review, gui, profiler, logger)

    classifing_keys = [Label.KEYWORD.key,
                       Label.NOT_RELEVANT.key,
                       Label.NOISE.key,
                       Label.RELEVANT.key]

    fawoc.add_key_binding(classifing_keys, lambda e: classify_kb(e, fawoc))
    fawoc.add_key_binding(['a'], lambda e: autonoise_kb(e, fawoc))
    fawoc.add_key_binding(['p'], lambda e: postpone_kb(e, fawoc))
    fawoc.add_key_binding(['u'], lambda e: undo_kb(e, fawoc))
    fawoc.add_key_binding(['w'], lambda e: save_kb(e, fawoc))
    fawoc.add_key_binding(['q'], quit_kb)
    fawoc.app.run()


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
    _, _ = terms.from_tsv(args.datafile)
    jsonfile = '.'.join([args.datafile, 'json'])
    try:
        terms.load_service_data(jsonfile)
    except FileNotFoundError:
        msg = f'File {jsonfile} not found. Service data not loaded.'
        debug_logger.info(msg)

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

    fawoc_main(terms, args, review, last_reviews, logger=debug_logger,
               profiler=profiler_logger)

    profiler_logger.info("CLASSIFIED: {}".format(terms.count_classified()))
    profiler_logger.info("DATAFILE '{}'".format(datafile_path))
    profiler_logger.info("*** PROGRAM TERMINATED ***")

    if review != Label.NONE:
        # ending review mode we must save some info
        last_reviews[datafile_path] = review.label_name

    if len(last_reviews) > 0:
        with open('last_review.json', 'w') as fout:
            json.dump(last_reviews, fout)

    if not args.dry_run:
        terms.to_tsv(args.datafile)
        terms.save_service_data(jsonfile)


if __name__ == "__main__":
    if DEBUG:
        input('Wait for debug...')
    main()
