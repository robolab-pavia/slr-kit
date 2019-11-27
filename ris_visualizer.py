import ast
import tkinter as tk
from tkinter import ttk
import pandas as pd
import utils


class TextWrapper(tk.Text):
    def __init__(self, master=None, cnf=None, **kwargs):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kwargs)

    def highlight_words(self, words):
        """
        Hightlights the specified words

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
            idx += list(utils.substring_index(s, w))
        for start, end in idx:
            sstart = '1.{}'.format(start)
            sstop = '1.{}'.format(end)
            self.tag_add('highlight', sstart, sstop)

    def highlight_style(self, color, bold=True):
        """
        Sets the highlight style

        :param color: string defining the color. Can be a name or a hexcode in the form '#RRGGBB'
        :type color: str
        :param bold: use bold style or not
        :type bold: bool
        """
        if bold:
            font = 'bold'
        else:
            font = ''

        self.tag_configure('highlight', font=font, foreground=color)


class StatusBar(tk.Frame):
    # Adapted from https://stackoverflow.com/a/8120427
    def __init__(self, master, height=8):
        tk.Frame.__init__(self, master)
        self.variable = tk.StringVar()
        self.label = tk.Label(self, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                              textvariable=self.variable,
                              font=('TkDefaultFont', f'{height}'))
        print(self.label['font'])
        self.variable.set(' ')
        self.label.pack(fill=tk.X)

    @property
    def text(self):
        return self.variable.get()

    @text.setter
    def text(self, new_text):
        self.variable.set(new_text)



class Gui:

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
        self.root = tk.Tk()
        self.root.title('RIS Visualizer')
        self.mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.mainframe_col_count = 0

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

        self.filterframe = ttk.LabelFrame(self.root, text='Filter')
        self.filterframe.grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.filter = self._setup_filter_entry()
        self.filter_box = self._setup_filter_combobox()

        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(column=0, row=4, sticky=(tk.W, tk.E))

        self._list_change_event(None)
        self.list_box.bind('<<ListboxSelect>>', self._list_change_event)
        self.list_box.focus()

    def _setup_filter_combobox(self):
        filter_box = ttk.Combobox(self.filterframe, state='readonly',
                                  values=('abstract', 'title'))
        filter_box.grid(row=0, column=0, sticky=(tk.E, tk.W))
        filter_box.set('abstract')
        return filter_box

    def _setup_filter_entry(self):
        """
        Setups the filter entry

        :return: the variable that contains the text of the filter entry
        :rtype: tk.StringVar
        """
        filter_var = tk.StringVar()
        fil = ttk.Entry(self.filterframe, textvariable=filter_var)
        fil.grid(row=0, column=1, sticky=(tk.W, tk.E))
        fil.grid_configure(padx=5, pady=5)
        fil.bind('<Key>', self._filter_set)
        return filter_var

    def _filter_set(self, event):
        """
        Method used as a callback for the Key event of the filter entry

        :param event: the event data
        :type event: tk.Event
        """
        if event.char == '\r':
            filter_txt = self.filter.get()
            if filter_txt == '':
                self.filter_txt = ''
                self.status_bar.text = ''
                if self.fdf is None:
                    return

                self.fdf = None
                df = self.df
            else:
                self.filter_field = self.filter_box.get()
                print(self.filter_field)

                cond = self._filter(filter_txt)
                s = f'Filter by {self.filter_field.upper()} Text: {filter_txt}'

                if any(cond):
                    self.fdf = self.df[cond]
                    df = self.fdf
                    self.filter_txt = filter_txt
                    s = f'{s} N° of results: {len(df)}'
                else:
                    self.fdf = None
                    df = self.df
                    self.filter_txt = ''
                    s = f'{s} No results'

                self.status_bar.text = s

            self.list_names.set(self._prepare_list(df))
            self.list_box.selection_set(first=0)
            self._list_change_event(None)
            self.list_box.focus()

    def _filter(self, filter_txt):
        def func(v):
            return utils.substring_check(v, filter_txt)

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

        # self.title.set(df['title'].iat[idx])
        self.title['state'] = 'normal'
        self.title.delete('1.0', 'end')
        self.title.insert('1.0', df['title'].iat[idx])
        self.title['state'] = 'disabled'
        self.abstract['state'] = 'normal'
        self.abstract.delete('1.0', 'end')
        self.abstract.insert('1.0', df['abstract'].iat[idx])
        self.abstract['state'] = 'disabled'

        if self.filter_field == 'abstract':
            self.abstract.highlight_words(self.filter_txt)
        elif self.filter_field == 'title':
            self.title.highlight_words(self.filter_txt)

        self.authors.set(df['authors'].iat[idx])
        self.year.set(df['year'].iat[idx])
        self.pub.set(df['secondary_title'].iat[idx])

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
        :return: the pubblication tk variable associated to the widget
        :rtype: tk.StringVar
        """
        ttk.Label(frame, text='Pubblication: ').grid(column=1, row=3,
                                                     sticky=(tk.W, tk.E))
        pub = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=pub)
        lbl.grid(column=2, row=3, sticky=(tk.W, tk.E))
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
        :return: the title tk variable associated to the widget
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


def main():
    df = pd.read_csv('rts-sample-ris.csv', sep='\t',
                     usecols=['id', 'authors', 'title', 'secondary_title',
                              'abstract', 'year'],
                     converters={'authors': authors_convert})

    gui = Gui(df)
    gui.root.mainloop()


if __name__ == '__main__':
    main()
