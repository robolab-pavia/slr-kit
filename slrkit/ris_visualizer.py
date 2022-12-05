import argparse
import ast
import string
import tkinter as tk
from tkinter import ttk
import pandas as pd
from utils import substring_index, substring_check

WORD_DELIMITERS = string.whitespace + ',;.:"\'()\\/-'


class SearchPanel(ttk.Frame):
    def __init__(self, row, column, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid(row=row, column=column, columnspan=2,
                  sticky=(tk.N, tk.W, tk.E, tk.S))
        self._filter_shown = False
        self._search_shown = False
        self._filterframe = ttk.LabelFrame(self, text='Filter')
        self.filter_txt, self.filter_entry = self._setup_filter_entry()
        self.filter_box = self._setup_filter_combobox()
        self.case_insensitive = self._setup_case_checkbutton()
        self._searchframe = ttk.LabelFrame(self, text='Search by ID')
        self.search_txt, self.search_entry = self._setup_search()

    @property
    def filter_shown(self):
        return self._filter_shown

    @property
    def search_shown(self):
        return self._search_shown

    def _setup_case_checkbutton(self):
        """
        Setups the case insensitive checkbutton

        :return: the variable associated to the checkbutton
        :rtype: tk.BooleanVar
        """
        case = tk.BooleanVar()
        case.set(False)
        check = ttk.Checkbutton(self._filterframe, text='Case insensitive',
                                variable=case, onvalue=True,
                                offvalue=False)
        check.grid(row=0, column=2, sticky=(tk.E, tk.W))
        return case

    def _setup_filter_combobox(self):
        """
        Setups the filter combobox
        """
        filter_box = ttk.Combobox(self._filterframe, state='readonly',
                                  values=('abstract', 'title', 'pubblication'))
        filter_box.grid(row=0, column=0, sticky=(tk.E, tk.W))
        filter_box.set('abstract')
        return filter_box

    def _setup_filter_entry(self):
        """
        Setups the filter entry

        :return: the variable that contains the text and the filter entry widget
        :rtype: (tk.StringVar, ttk.Entry)
        """
        filter_var = tk.StringVar()
        fil = ttk.Entry(self._filterframe, textvariable=filter_var)
        fil.grid(row=0, column=1, sticky=(tk.W, tk.E))
        fil.grid_configure(padx=5, pady=3)
        return filter_var, fil

    def _setup_search(self):
        search = tk.StringVar()
        search_entry = ttk.Entry(self._searchframe, textvariable=search)
        search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        search_entry.grid_configure(padx=5, pady=3)
        return search, search_entry

    def toggle_filter(self):
        if self._filter_shown:
            self._filterframe.grid_remove()
            if self._search_shown:
                self._searchframe.grid_remove()
                self._searchframe.grid(row=0, column=0, sticky=(tk.W, tk.E))

        else:
            self._filterframe.grid(row=0, column=0, sticky=(tk.W, tk.E))
            if self._search_shown:
                self._searchframe.grid_remove()
                self._searchframe.grid(row=0, column=1, sticky=(tk.W, tk.E))

        self._filter_shown = not self._filter_shown

    def toggle_search(self):
        if self._search_shown:
            self._searchframe.grid_remove()

        else:
            if self._filter_shown:
                self._searchframe.grid(row=0, column=1, sticky=(tk.W, tk.E))
            else:
                self._searchframe.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self._search_shown = not self._search_shown


class TextWrapper(tk.Text):
    """
    Wrapper class for the Text widget
    """

    def __init__(self, master=None, cnf=None, **kwargs):
        if cnf is None:
            cnf = {}

        super().__init__(master, cnf, **kwargs)

    def set(self, text):
        """
        Sets the text of the TextWrapper

        :param text: text to show
        :type text: str
        """
        disabled = False
        if self['state'] == 'disabled':
            disabled = True
            self['state'] = 'normal'

        self.delete('1.0', 'end')
        self.insert('1.0', text)
        if disabled:
            self['state'] = 'disabled'

    def highlight_words(self, words, case_insensitive=False):
        """
        Hightlights the specified words

        :param case_insensitive: if the word search must be case insensitive (Default: False)
        :type case_insensitive: bool
        :param words: list of words to highlight
        :type words: str or list[str]
        """
        tags = self.tag_ranges('highlight')
        for i in range(0, len(tags) - 1, 2):
            self.tag_remove('highlight', tags[i], tags[i + 1])

        if isinstance(words, str):
            words = [words]

        s = self.get('1.0', 'end')
        idx = []
        for w in words:
            if case_insensitive:
                s = s.lower()
                w = w.lower()

            idx += list(substring_index(s, w, delim=WORD_DELIMITERS))

        for start, end in idx:
            sstart = '1.{}'.format(start)
            sstop = '1.{}'.format(end)
            self.tag_add('highlight', sstart, sstop)

    def highlight_style(self, color, bold=True):
        """
        Sets the highlight style

        :param color: color to use. Can be a name or a hexcode in the form '#RRGGBB'
        :type color: str
        :param bold: use bold style or not
        :type bold: bool
        """
        if bold:
            font = 'bold'
        else:
            font = ''

        self.tag_configure('highlight', font=font, foreground=color)


# Adapted from https://stackoverflow.com/a/8120427
class StatusBar(tk.Frame):
    """
    Status bar widget
    """

    def __init__(self, master, height=8):
        """
        Creates a new StatusBar

        :param master: the master widget of the StatusBar
        :type master: tk.Widget
        :param height: height of the StatusBar in font points
        :type height: int
        """
        tk.Frame.__init__(self, master)
        self.variable = tk.StringVar()
        self.label = tk.Label(self, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                              textvariable=self.variable,
                              font=('TkDefaultFont', f'{height}'))
        self.variable.set(' ')
        self.label.pack(fill=tk.X)

    @property
    def text(self):
        """
        Text of the status bar
        """
        return self.variable.get()

    @text.setter
    def text(self, new_text):
        self.variable.set(new_text)


class Gui:
    """
    Gui of the application

    :type fdf: pd.DataFrame or None
    """

    def __init__(self, df):
        """
        Creates the GUI of the app

        :param df: dataframe of the RIS data
        :type df: pd.DataFrame
        """
        self.df = df
        self.filter_txt = ''
        self.fdf = None
        self.filter_field = ''
        self.filter_case_insensitive = False
        self.root = tk.Tk()
        self.root.title('RIS Visualizer')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.root.bind('<Key>', self._key_press)

        # main part of the window
        self._setup_mainframe()

        # filter part
        self._setup_filter()
        self.filterpanel.filter_entry.bind('<Key>', self._filter_set)
        self.filterpanel.search_entry.bind('<Key>', self._search_set)

        # status bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(column=0, row=4, columnspan=2, sticky=(tk.W, tk.E))

        # events
        self._list_change_event(None)
        self.list_box.bind('<<ListboxSelect>>', self._list_change_event)

        self.list_box.focus()

    def _search_set(self, event):
        if getattr(event, 'char', '') == '\r':
            txt = self.filterpanel.search_txt.get()
            if txt != '':
                try:
                    search_id = int(txt)
                except ValueError:
                    search_id = -1

                if 0 <= search_id < len(self.df):
                    self.list_box.select_clear(0, tk.END)
                    self.list_box.select_set(first=search_id)
                    self.list_box.see(search_id)
                    self.list_box.activate(search_id)
                    self._list_change_event(None)
                else:
                    s = self.status_bar.text
                    if len(s) > 0:
                        s = f'{s}         '

                    s = f'{s}Invalid id: {txt}'
                    self.status_bar.text = s

            self.list_box.focus()

    def _key_press(self, event):
        entries = [self.filterpanel.filter_entry, self.filterpanel.search_entry]
        if self.root.focus_get() not in entries:
            key = getattr(event, 'char', '').lower()
            if key == 'f':
                self.filterpanel.toggle_filter()
            elif key == 's':
                self.filterpanel.toggle_search()

            if (not self.filterpanel.filter_shown
                    and not self.filterpanel.search_shown):
                self.filterpanel.grid_remove()
            else:
                self.filterpanel.grid()

    def _setup_filter(self):
        """
        Setups the whole filter frame of the application
        """
        self.filterpanel = SearchPanel(1, 0, master=self.root)

    def _setup_mainframe(self):
        """
        Setups the whole mainframe of the application
        """
        self.mainframe_col_count = 0
        self.mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, columnspan=2,
                            sticky=(tk.N, tk.W, tk.E, tk.S))
        self.list_names, self.list_box = self._setup_list(self.mainframe)
        self.title = self._setup_title(self.mainframe)
        self.abstract = self._setup_abstract()
        other_frame = ttk.Frame(self.mainframe)
        other_frame.grid(column=3, row=3, columnspan=2,
                         sticky=(tk.N, tk.W, tk.E, tk.S))
        self.authors = self._setup_authors(other_frame)
        self.year = self._setup_year(other_frame)
        self.pub = self._setup_pubblication(other_frame)
        for child in self.mainframe.winfo_children():
            if any(s in str(child) for s in ['listbox', 'text', 'scrollbar']):
                continue

            child.grid_configure(padx=5, pady=5)

    def _filter_set(self, event):
        """
        Method used as a callback for the Key event of the filter entry

        :param event: the event data
        :type event: tk.Event
        """
        if getattr(event, 'char', '') == '\r':
            filter_txt = self.filterpanel.filter_txt.get()
            self.filter_case_insensitive = self.filterpanel.case_insensitive.get()
            if filter_txt == '':
                self.filter_txt = ''
                self.status_bar.text = ''
                if self.fdf is None:
                    return

                self.fdf = None
                df = self.df
            else:
                self.filter_field = self.filterpanel.filter_box.get()
                print(self.filter_field)

                self.filter_txt = filter_txt
                cond = self._filter(self.filter_case_insensitive)
                s = f'Filter by {self.filter_field.upper()} Text: {filter_txt}'

                if any(cond):
                    self.fdf = self.df[cond]
                    df = self.fdf
                    s = f'{s} N° of results: {len(df)}'
                else:
                    self.fdf = None
                    df = self.df
                    self.filter_txt = ''
                    s = f'{s} No results'

                self.status_bar.text = s
                print(self.status_bar.text)

            self.list_names.set(self._prepare_list(df))
            self.list_box.selection_set(first=0)
            self._list_change_event(None)
            self.list_box.focus()

    def _filter(self, case_insensitive=False):
        """
        Filters the dataframe using the info in self.filter_txt and self.filter_field

        :param case_insensitive: if the search is case insensitive or not (Default: False)
        :type case_insensitive: bool
        :return: a Series indicating which rows of self.df correspond to the filter
        :rtype: pd.Series
        """
        delim = WORD_DELIMITERS

        def func(v):
            if case_insensitive:
                v = v.lower()
                txt = self.filter_txt.lower()
            else:
                txt = self.filter_txt

            return substring_check(v, txt, delim=delim)

        cond = self.df[self.filter_field].apply(func)
        return cond

    def _list_change_event(self, event):
        """
        Method used as a callback for the ListboxSelect event

        :param event: the event data or None if this method is not called by Tk
        :type event: tk.Event or None
        """
        try:
            idx = self.list_box.curselection()[0]
        except IndexError:
            return

        if self.fdf is not None:
            df = self.fdf
        else:
            df = self.df

        self.title.set(df['title'].iat[idx])
        self.abstract.set(df['abstract'].iat[idx])
        self.pub.set(df['pubblication'].iat[idx])

        if self.filter_field == 'abstract':
            self.abstract.highlight_words(self.filter_txt,
                                          self.filter_case_insensitive)
        elif self.filter_field == 'title':
            self.title.highlight_words(self.filter_txt,
                                       self.filter_case_insensitive)
        elif self.filter_field == 'pubblication':
            self.pub.highlight_words(self.filter_txt,
                                     self.filter_case_insensitive)

        self.authors.set(df['authors'].iat[idx])
        self.year.set(df['year'].iat[idx])

    def _setup_list(self, frame):
        """
        Setups the list of papers

        :param frame: frame where to put the list widget
        :type frame: ttk.Frame
        :return: the list variable to modify the list and the listbox widget
        :rtype: (tk.StringVar, tk.Listbox)
        """
        lst = self._prepare_list(self.df)

        list_names = tk.StringVar(value=lst)
        list_box = tk.Listbox(frame, height=10, width=40,
                              listvariable=list_names, selectmode='browse')
        self.mainframe_col_count += 1
        list_box.grid(column=self.mainframe_col_count, row=1, rowspan=6,
                      sticky=(tk.N, tk.W, tk.E, tk.S), padx=(5, 0))
        list_box.selection_set(first=0)
        list_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL,
                                      command=list_box.yview)
        self.mainframe_col_count += 1
        list_scrollbar.grid(column=self.mainframe_col_count, row=1, rowspan=6,
                            sticky=(tk.N, tk.S), padx=(0, 5))
        list_box['yscrollcommand'] = list_scrollbar.set
        return list_names, list_box

    @staticmethod
    def _prepare_list(df):
        """
        Creates the list of entry to be used in the listbox

        :param df: Dataframe from which to take the data
        :type df: pd.DataFrame
        :return: the prepared list
        :rtype: list[str]
        """
        lst: list = df.apply(lambda r: '{} - {}'.format(r['id'], r['title']),
                             axis=1).tolist()
        return lst

    @staticmethod
    def _setup_pubblication(frame):
        """
        Setups the pubblication widget and label

        :param frame: frame where to put the pubblication widget and label
        :type frame: ttk.Frame
        :return: the pubblication widget
        :rtype: TextWrapper
        """
        ttk.Label(frame, text='Pubblication: ').grid(column=1, row=3,
                                                     sticky=(tk.W, tk.E))
        pub = TextWrapper(frame, wrap='word', state='disabled', height=1)
        pub['relief'] = 'flat'
        pub['borderwidth'] = 0
        pub['highlightthickness'] = 0
        pub['background'] = '#d9d9d9'
        pub['foreground'] = '#000000'
        pub['font'] = 'TkDefaultFont'
        pub.highlight_style('red')
        pub.grid(column=2, row=3, sticky=(tk.W, tk.E))
        return pub

    @staticmethod
    def _setup_year(frame):
        """
        Setups the year widget and label

        :param frame: frame where to put the year widget and label
        :type frame: ttk.Frame
        :return: the year tk variable associated to the widget
        :rtype: tk.StringVar
        """
        ttk.Label(frame, text='Year: ').grid(column=1, row=2,
                                             sticky=(tk.W, tk.E))
        year = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=year)
        lbl.grid(column=2, row=2, sticky=(tk.W, tk.E))
        return year

    @staticmethod
    def _setup_authors(frame):
        """
        Setups the authors widget and label

        :param frame: frame where to put the author widget and label
        :type frame: ttk.Frame
        :return: the author tk variable associated to the widget
        :rtype: tk.StringVar
        """
        ttk.Label(frame, text='Authors: ').grid(column=1, row=1,
                                                sticky=(tk.W, tk.E))
        authors = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=authors)
        lbl.grid(column=2, row=1, sticky=(tk.W, tk.E))
        return authors

    def _setup_abstract(self):
        """
        Setups the text widget for the abstract.

        It setups also the vertical scrollbar
        :return: the text widget
        :rtype: TextWrapper
        """
        abstract = TextWrapper(self.mainframe, wrap='word', state='disabled',
                               height=10)
        self.mainframe_col_count += 1
        abstract.grid(column=self.mainframe_col_count, row=2, columnspan=2,
                      sticky=(tk.W, tk.E))
        abs_scrollbar = tk.Scrollbar(self.mainframe, orient=tk.VERTICAL,
                                     command=abstract.yview)
        self.mainframe_col_count += 2
        abs_scrollbar.grid(column=self.mainframe_col_count, row=2,
                           sticky=(tk.N, tk.S))
        abstract['yscrollcommand'] = abs_scrollbar.set
        abstract.highlight_style('red')
        return abstract

    @staticmethod
    def _setup_title(frame):
        """
        Setups the paper title

        :param frame: frame where to put the title widget
        :type frame: ttk.Frame
        :return: the title widget
        :rtype: TextWrapper
        """
        # title = tk.StringVar()
        # lbl = ttk.Label(frame, textvariable=title)
        # lbl.grid(column=3, row=1, sticky=(tk.W, tk.E))
        title = TextWrapper(frame, wrap='word', state='disabled', height=1)
        title.grid(column=3, row=1, sticky=(tk.W, tk.E))
        title['relief'] = 'flat'
        title['borderwidth'] = 0
        title['highlightthickness'] = 0
        title['background'] = '#d9d9d9'
        title['foreground'] = '#000000'
        title['font'] = 'TkDefaultFont'
        title.highlight_style('red')
        return title


def authors_convert(auth_str):
    """
    Converts the string in a list of authors

    :param auth_str: the string to convert. Must be a valid python list declaration
    :type auth_str: str
    :return: the converted list
    :rtype: list[str]
    """
    auth_list = ast.literal_eval(auth_str)
    authors = ', '.join([a.replace(',', '') for a in auth_list])
    return authors


def init_argparser():
    """
    Initialize the command line parser.

    :return: the command line parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', action="store", type=str,
                        help="input TSV data file")
    return parser


def usecols(col):
    """
    Helper function to select columns in input file

    :param col: the name of the column to evaluate
    :type col: str
    :return: True if col must be included and False otherwise
    :rtype: bool
    """
    colnames = ['id', 'authors', 'title', 'secondary_title', 'abstract',
                'abstract1', 'year']
    return col in colnames


def prepare_df(args):
    """
    Loads and prepare the dataframe with information about the papers

    :param args: command line arguments
    :type args: argparse.Namespace
    :return: the loaded dataframe
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(args.datafile, sep='\t',
                     usecols=usecols, encoding='utf-8',
                     converters={'authors': authors_convert})
    df.rename(columns={'secondary_title': 'pubblication',
                       'abstract1': 'abstract'}, inplace=True)
    for f in ['id', 'abstract']:
        if f not in df.columns:
            raise ValueError(f'Missing required field {f} in {args.datafile}')

    for f in ['authors', 'title', 'year', 'pubblication']:
        if f not in df.columns:
            df[f] = [''] * len(df)

    df.fillna('', inplace=True)

    return df


def main():
    parser = init_argparser()
    args = parser.parse_args()
    df = prepare_df(args)

    gui = Gui(df)
    gui.root.mainloop()


if __name__ == '__main__':
    main()
