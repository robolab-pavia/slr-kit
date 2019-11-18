import ast
import tkinter as tk
from tkinter import ttk
import pandas as pd


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

        self.list_names, self.list_box = self._setup_list(self.mainframe)
        self.title = self._setup_title(self.mainframe)
        self.abstract = self._setup_abstract(self.mainframe)
        other_frame = ttk.Frame(self.mainframe)
        other_frame.grid(column=3, row=3, columnspan=2,
                         sticky=(tk.N, tk.W, tk.E, tk.S))
        self.authors = self._setup_authors(other_frame)
        self.year = self._setup_year(other_frame)
        self.pub = self._setup_pubblication(other_frame)

        for child in self.mainframe.winfo_children():
            if any(s in str(child) for s in ['text', 'scrollbar']):
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
        lst = self.df.apply(lambda r: '{} - {}'.format(r['id'], r['title']),
                            axis=1)
        lst = lst.to_list()
        list_names = tk.StringVar(value=lst)
        list_box = tk.Listbox(frame, height=10,
                              listvariable=list_names, selectmode='browse')
        list_box.grid(column=1, row=1, rowspan=6, sticky=(tk.N, tk.W,
                                                          tk.E, tk.S))
        list_box.selection_set(first=0)
        return list_names, list_box

    @staticmethod
    def _setup_pubblication(frame):
        ttk.Label(frame, text='Pubblication: ').grid(column=1, row=3,
                                                     sticky=(tk.W, tk.E))
        pub = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=pub)
        lbl.grid(column=2, row=3, sticky=(tk.W, tk.E))
        return pub

    @staticmethod
    def _setup_year(frame):
        ttk.Label(frame, text='Year: ').grid(column=1, row=2,
                                             sticky=(tk.W, tk.E))
        year = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=year)
        lbl.grid(column=2, row=2, sticky=(tk.W, tk.E))
        return year

    @staticmethod
    def _setup_authors(frame):
        ttk.Label(frame, text='Authors: ').grid(column=1, row=1,
                                                sticky=(tk.W, tk.E))
        authors = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=authors)
        lbl.grid(column=2, row=1, sticky=(tk.W, tk.E))
        return authors

    @staticmethod
    def _setup_abstract(frame):
        abstract = tk.Text(frame, wrap='word', state='disabled',
                           height=10)
        abstract.grid(column=3, row=2, columnspan=2, sticky=(tk.W, tk.E))
        abs_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL,
                                     command=abstract.yview)
        abs_scrollbar.grid(column=5, row=2, sticky=(tk.N, tk.S))
        abstract['yscrollcommand'] = abs_scrollbar.set
        return abstract

    @staticmethod
    def _setup_title(frame):
        title = tk.StringVar()
        lbl = ttk.Label(frame, textvariable=title)
        lbl.grid(column=3, row=1, sticky=(tk.W, tk.E))
        return title


def authors_convert(auth_str):
    """

    :param auth_str:
    :type auth_str: str
    :return:
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
