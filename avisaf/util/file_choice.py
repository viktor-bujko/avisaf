#!/usr/bin/env python3

import pandas as pd
import tkinter
from tkinter import filedialog
from sys import stderr


def _choose_file(select=True):
    """

    :return:
    """
    if select:
        tkinter.Tk()
        file_path = filedialog.askopenfilename(initialdir='~', title='Select a file', filetypes=[('csv files', '*.csv')])

        return file_path
    else:
        return '/media/sf_MFF_Skola/2rocnik/Rocnikovy-projekt/data/ASRS/ASRS-csv-reports/ASRS_DBOnline-04-2019-12-2019.csv'


def extract_narratives(lines=-1, file_path=None, start_index=0):
    """

    :param lines:
    :param file_path:
    :param start_index:
    :return:
    """

    if file_path is None:
        ASRS_file_path = _choose_file()
    else:
        ASRS_file_path = file_path

    report_df = pd.read_csv(ASRS_file_path, skip_blank_lines=True, index_col=0, header=[0, 1])
    report_df.columns = report_df.columns.map('_'.join)

    try:
        narrs1 = report_df['Report 1_Narrative'].values.tolist()
        calls1 = report_df['Report 1_Callback'].values.tolist()
        narrs2 = report_df['Report 2_Narrative'].values.tolist()
        calls2 = report_df['Report 2_Callback'].values.tolist()
        length = len(narrs1)
        lists = [narrs1, calls1, narrs2, calls2]

        # assert all(len(lst) == length for lst in lists)
        for index in range(start_index, length):
            if lines != -1 and index >= lines:
                break
            res = '\n'.join([str(lst[index]) for lst in lists if str(lst[index]) != 'nan'])
            yield res

    except KeyError:
        print('No such key was found', file=stderr)
        return None
