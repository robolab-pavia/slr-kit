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


class Gui:

    def __init__(self, df):
        """
        Creates the GUI of the app

        :param df: dataframe of the RIS data
        :type df: pd.DataFrame
        """
        self.df = df
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

        self._list_change_event(None)
        self.list_box.bind('<<ListboxSelect>>', self._list_change_event)
        self.list_box.focus()

    def _list_change_event(self, event):
        try:
            idx = self.list_box.curselection()[0]
        except IndexError:
            return

        self.title.set(self.df.loc[idx, 'title'])
        self.abstract['state'] = 'normal'
        self.abstract.delete('1.0', 'end')
        self.abstract.insert('1.0', self.df.loc[idx, 'abstract'])
        self.abstract['state'] = 'disabled'
        self.authors.set(self.df.loc[idx, 'authors'])
        self.year.set(self.df.loc[idx, 'year'])
        self.pub.set(self.df.loc[idx, 'secondary_title'])

    def _setup_list(self, frame):
        """
        Setups the list of papers

        :param frame: frame where to put the list widget
        :type frame: ttk.Frame
        :return: the list variable to modify the list and the listbox widget
        :rtype: (tk.StringVar, tk.Listbox)
        """
        lst: list = self.df.apply(lambda r: '{} - {}'.format(r['id'],
                                                             r['title']),
                                  axis=1).tolist()

        list_names = tk.StringVar(value=lst)
        list_box = tk.Listbox(frame, height=10,
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
        :rtype: tk.Text
        """
        abstract = tk.Text(self.mainframe, wrap='word', state='disabled',
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
        return abstract

    @staticmethod
    def _setup_title(frame):
        """
        Setups the paper title

        :param frame: frame where to put the title widget
        :type frame: ttk.Frame
        :return: the title tk variable associated to the widget
        :rtype: tk.StringVar
        """
        title = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=title)
        lbl.grid(column=3, row=1, sticky=(tk.W, tk.E))
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
