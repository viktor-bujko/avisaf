#!/usr/bin/env python3
"""The data extractor module is responsible for getting raw text data and
training data used by other modules.
"""

import pandas as pd
from sys import stderr
import json
import sys
from pathlib import Path

# looking for the project root
path = Path(__file__)
while not str(path.resolve()).endswith('avisaf_ner'):
    path = path.parent.resolve()

SOURCES_ROOT_PATH = Path(path, 'avisaf').resolve()
PROJECT_ROOT_PATH = path.resolve()
sys.path.append(str(SOURCES_ROOT_PATH))


def get_entities(entities_file_path: Path = Path(PROJECT_ROOT_PATH, 'entities_labels.json').resolve()):
    """Function which reads given JSON file supposed to contain the list of user
    defined entity labels.

    :type entities_file_path: Path
    :param entities_file_path: The path to the JSON file containing the list
        of entity labels.

    :return: Returns the list of available entity labels.
    """
    entities_file_path = entities_file_path.resolve()

    with entities_file_path.open(mode='r') as entities_file:
        return json.load(entities_file)


def get_training_data(training_data_file_path: Path):
    """Function which reads given JSON file supposed to contain the training data.
    The training data are supposed to be a list of (text, annotations) tuples.

    :type training_data_file_path: Path
    :param training_data_file_path: The path to the JSON file containing the
        training data.

    :return: Returns the JSON list of (text, annotations) tuples.
    """

    if not training_data_file_path.is_absolute():
        training_data_file_path = training_data_file_path.resolve()

    with training_data_file_path.open(mode='r') as tr_data_file:
        return json.load(tr_data_file)


def get_narratives(lines: int = -1, file_path: Path = None, start_index: int = 0):
    """Function responsible for reading raw csv file containing the original
    safety reports from the ASRS database.

    :type lines: int
    :param lines: Number of lines to be read.
    :type file_path: Path
    :param file_path: The path to the csv file containing the texts.
    :type start_index: int
    :param start_index: Number indicating the index of the first text to be
        returned.

    :return: Returns a python generator object of all texts.
    """

    file_path = str(file_path) if file_path.is_absolute() else str(file_path.resolve())

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
        res = ' '.join([str(lst[index]) for lst in lists if str(lst[index]) != 'nan'])
        yield res
