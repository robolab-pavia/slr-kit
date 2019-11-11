import ast
import tkinter as tk
from tkinter import ttk
import pandas as pd


def change_selection(event, df, tk_vars):
    idx = event.widget.curselection()[0]
    tk_vars['title'].set(df.loc[idx, 'title'])
    tk_vars['abstract']['state'] = 'normal'
    tk_vars['abstract'].insert('1.0', df.loc[idx, 'abstract'])
    tk_vars['abstract']['state'] = 'disabled'
    tk_vars['authors'].set(df.loc[idx, 'authors'])


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

    lst = df.apply(lambda r: '{} - {}'.format(r['id'], r['title']), axis=1)
    lst = lst.to_list()
    root = tk.Tk()
    root.title('RIS Visualizer')
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E,
                                            tk.S))
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    lnames = tk.StringVar(value=lst)
    lbox = tk.Listbox(mainframe, height=10, listvariable=lnames,
                      selectmode='browse')
    lbox.grid(column=1, row=1, rowspan=6, sticky=(tk.N, tk.W,
                                                  tk.E, tk.S))
    lbox.selection_set(first=0)

    title = tk.StringVar()
    ttk.Label(mainframe, textvariable=title).grid(column=3, row=1,
                                                  sticky=(tk.W, tk.E))

    abstract = tk.Text(mainframe, wrap='word', state='disabled', height=10)
    abstract.grid(column=3, row=2, sticky=(tk.W, tk.E))
    abs_scrollbar = tk.Scrollbar(mainframe, orient=tk.VERTICAL,
                                 command=abstract.yview)
    abs_scrollbar.grid(column=4, row=2, sticky=(tk.N, tk.S))
    abstract['yscrollcommand'] = abs_scrollbar.set

    authors = tk.StringVar()
    ttk.Label(mainframe, textvariable=authors).grid(column=3, row=3,
                                                    sticky=(tk.W, tk.E))
    title.set(df.loc[0, 'title'])
    abstract['state'] = 'normal'
    abstract.insert('1.0', df.loc[0, 'abstract'])
    abstract['state'] = 'disabled'
    authors.set(df.loc[0, 'authors'])

    tk_vars = {'title': title, 'abstract': abstract, 'authors': authors}

    lbox.bind('<<ListboxSelect>>',
              lambda event: change_selection(event, df, tk_vars))
    root.mainloop()


if __name__ == '__main__':
    main()
