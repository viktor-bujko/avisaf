#!/usr/bin/env python3

import pandas as pd
import tkinter
from tkinter import filedialog
from sys import stderr
import os
import json


def get_entities(entities_file_path=os.path.expanduser('~/Documents/avisaf_ner/resources/entities_labels.json')):
    """

    :param entities_file_path:
    :return:
    """
    with open(entities_file_path, mode='r') as entities_file:
        return json.load(entities_file)


def get_training_data(training_data_file_path):
    """

    :return:
    """
    with open(training_data_file_path, mode='r') as tr_data_file:
        return json.load(tr_data_file)


def _choose_file(select=False):
    """

    :return:
    """
    if select:
        tkinter.Tk()
        file_path = filedialog.askopenfilename(initialdir='~', title='Select a file', filetypes=[('csv files', '*.csv')])

        return file_path
    else:
        return '/media/sf_MFF_Skola/2rocnik/Rocnikovy-projekt/data/ASRS/ASRS-csv-reports/ASRS_DBOnline-04-2019-12-2019.csv'


def get_narratives(lines=-1, file_path=None, start_index=0):
    """
    :type lines:
    :param lines:
    :type file_path:
    :param file_path:
    :type start_index:
    :param start_index:
    :return: Returns a generator of all texts.
    """

    if file_path is None:
        file_path = _choose_file()

    report_df = pd.read_csv(file_path, skip_blank_lines=True, index_col=0, header=[0, 1])
    report_df.columns = report_df.columns.map('_'.join)

    try:
        narratives1 = report_df['Report 1_Narrative'].values.tolist()
        calls1 = report_df['Report 1_Callback'].values.tolist()
        narratives2 = report_df['Report 2_Narrative'].values.tolist()
        calls2 = report_df['Report 2_Callback'].values.tolist()

    except KeyError:
        print('No such key was found', file=stderr)
        return None

    length = len(narratives1)
    lists = [narratives1, calls1, narratives2, calls2]

    # assert all(len(lst) == length for lst in lists)
    end_index = start_index + lines
    for index in range(start_index, length):
        if lines != -1 and index >= end_index:
            break
        res = '\n'.join([str(lst[index]) for lst in lists if str(lst[index]) != 'nan'])
        yield res
