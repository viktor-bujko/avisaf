#!/usr/bin/env python3

import pandas as pd
import tkinter
from tkinter import filedialog
from sys import stderr


def choose_file():
    tkinter.Tk()
    file_path = filedialog.askopenfilename(initialdir='~', title='Select a file',
                                           filetypes=[('csv files', '*.csv')])
    return file_path


def extract_narratives(csv_file_path='/media/sf_MFF_Skola/2rocnik/Rocnikovy-projekt/data/ASRS/ASRS-csv-reports/ASRS_DBOnline-01-2020-08-2020.csv'):

    report_df = pd.read_csv(csv_file_path, skip_blank_lines=True, index_col=0, header=[0, 1])
    report_df.columns = report_df.columns.map('_'.join)
    try:
        narrs1 = report_df['Report 1_Narrative'].values.tolist()
        calls1 = report_df['Report 1_Callback'].values.tolist()
        narrs2 = report_df['Report 2_Narrative'].values.tolist()
        calls2 = report_df['Report 2_Callback'].values.tolist()
        length = len(narrs1)
        lists = [narrs1, calls1, narrs2, calls2]
        # result = []
        assert all(len(lst) == length for lst in lists)
        for index in range(length):
            res = '\n'.join([str(lst[index]) for lst in lists if str(lst[index]) != 'nan'])
            yield res
        # return result
    except KeyError:
        print('No such key was found', file=stderr)


if __name__ == '__main__':
    # for narr in extract_narratives():
    #    print(narr + '\n\n\n')
    for value in  extract_narratives():
        print(value)
    # extract_narratives(choose_file())
