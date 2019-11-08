import tkinter as tk
from tkinter import ttk
import pandas as pd


def change_selection(event, df, tk_vars):
    idx = event.widget.curselection()[0]
    tk_vars['title'].set(df.loc[idx, 'title'])


def main():
    df = pd.read_csv('rts-sample-ris.csv', sep='\t',
                     usecols=['id', 'authors', 'title', 'secondary_title',
                              'abstract', 'year'])
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
    tlbl = ttk.Label(mainframe, text='Title:')
    tlbl.grid(column=2, row=1, sticky=(tk.W,))

    title = tk.StringVar()
    ttext = ttk.Label(mainframe, textvariable=title)
    ttext.grid(column=3, row=1, sticky=(tk.W, tk.E))
    title.set(df.loc[0, 'title'])

    tk_vars = {'title': title}

    lbox.bind('<<ListboxSelect>>', lambda event: change_selection(event, df, tk_vars))
    root.mainloop()


if __name__ == '__main__':
    main()
