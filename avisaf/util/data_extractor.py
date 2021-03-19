#!/usr/bin/env python3
"""The data extractor module is responsible for getting raw text data and
training data used by other modules.
"""

import pandas as pd
import sys
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format=f'[%(levelname)s - %(asctime)s]: %(message)s'
)


class JsonDataExtractor:
    # TODO: This class should contain methods for working with json files
    # get_entities, get_training_data
    pass


def get_entities(entities_file_path: Path = None):
    # Works with entities_labels.json file = only a simple json list
    # Probably will be moved to JsonDataExtractor
    """Function which reads given JSON file supposed to contain the list of user
    defined entity labels.

    :type entities_file_path: Path
    :param entities_file_path: The path to the JSON file containing the list
        of entity labels.

    :return: Returns the list of available entity labels.
    """
    if entities_file_path is None:
        # entities_file_path = Path(Path().resolve(), 'avisaf_ner/entities_labels.json')
        entities_file_path = find_file_by_path('entities_labels.json')

    entities_file_path = entities_file_path.resolve()

    with entities_file_path.open(mode='r') as entities_file:
        return json.load(entities_file)


def get_training_data(training_data_file_path: Path):
    # Works with (text, annotations) list JSON file
    # Probably will be moved to JsonDataExtractor
    """Function which reads given JSON file supposed to contain the training data.
    The training data are supposed to be a list of (text, annotations) tuples.

    :type training_data_file_path: Path
    :param training_data_file_path: The path to the JSON file containing the
        training data.

    :return: Returns the JSON list of (text, annotations) tuples.
    """
    if not training_data_file_path:
        msg = 'Training data file path cannot be None'
        logging.error(msg)
        raise TypeError(msg)

    if not training_data_file_path.is_absolute():
        training_data_file_path = training_data_file_path.resolve()

    with training_data_file_path.open(mode='r') as tr_data_file:
        return json.load(tr_data_file)


def get_narratives(file_path: Path, lines_count: int = -1, start_index: int = 0):
    # TODO: This method should be replaced by CsvDataExtractor.extract_data_from_csv_columns

    """Function responsible for reading raw csv file containing the original
    safety reports from the ASRS database.

    :type lines_count: int
    :param lines_count: Number of lines to be read.
    :type file_path: Path
    :param file_path: The path to the csv file containing the texts.
    :type start_index: int
    :param start_index: Number indicating the index of the first text to be
        returned.

    :return: Returns a python generator object of all texts.
    """

    if file_path is None:
        msg = 'The file_path to have narratives extracted from cannot be None'
        logging.error(msg)
        raise TypeError(msg)

    file_path = str(file_path) if file_path.is_absolute() else str(file_path.resolve())

    report_df = pd.read_csv(
        file_path,
        skip_blank_lines=True,
        index_col=0,
        header=[0, 1]
    )
    report_df.columns = report_df.columns.map('_'.join)

    try:
        narratives1 = report_df['Report 1_Narrative'].values.tolist()
        calls1 = report_df['Report 1_Callback'].values.tolist()
        narratives2 = report_df['Report 2_Narrative'].values.tolist()
        calls2 = report_df['Report 2_Callback'].values.tolist()

    except KeyError:
        print('No such key was found', file=sys.stderr)
        return None

    length = len(narratives1)
    lists = [narratives1, calls1, narratives2, calls2]

    end_index = start_index + lines_count

    result = []
    for index in range(start_index, length):
        if lines_count != -1 and index >= end_index:
            break
        result.append(' '.join([str(lst[index]) for lst in lists if str(lst[index]) != 'nan']))

    return result


def find_file_by_path(file_path: [Path, str], max_iterations: int = 5):
    # Static utility function not tied to any object
    """

    :param max_iterations:
    :param file_path:
    :return:
    """

    current_dir = Path().resolve()

    for iteration in range(max_iterations):  # Searching at most max_iterations times.
        if current_dir.parent is None or current_dir.parent == current_dir:
            break

        requested_file = current_dir.joinpath(Path(file_path)).resolve()

        if requested_file.exists():
            return requested_file
        else:
            current_dir = current_dir.parent

    return None


class DataExtractor:

    def __init__(self, file_paths: list):
        self._file_paths = file_paths

    def extract_from_csv_columns(self, field_name: [str, list], lines_count: int = -1,
                                 start_index: int = 0, file_paths: [list, str] = None):
        """

        :param file_paths:
        :param field_name:
        :param lines_count:
        :param start_index:
        :return:
        """

        # Normalizing the type of arguments to fit the rest of the method
        field_names = field_name if type(field_name) is list else [field_name]

        # Overriding default instance file paths list by the passed parameter
        file_paths = self._file_paths if file_paths is None else (file_paths if file_paths is list else [file_paths])

        label_data_dict = {}
        for a_field_name in field_names:
            extracted_values = []
            for a_file_path in file_paths:
                requested_file = find_file_by_path(a_file_path)
                if requested_file is None:
                    print(f'The file given by "{a_file_path}" path was not found in the given range.', file=sys.stderr)
                    # Ignoring the file with current file path
                    continue

                csv_dataframe = pd.read_csv(
                    requested_file,
                    skip_blank_lines=True,
                    header=[0, 1],
                )
                csv_dataframe.columns = csv_dataframe.columns.map('_'.join)
                csv_dataframe = csv_dataframe.replace(np.nan, "", regex=True)

                try:
                    extracted_values_collection = csv_dataframe[a_field_name].values.tolist()
                except KeyError:
                    print(
                        f'"{field_name} is not a correct field name. Please make sure the column name is in format "FirstLineTitle_SecondLineTitle"',
                        file=sys.stderr
                    )
                    continue

                length = len(extracted_values_collection)
                end_index = start_index + lines_count if lines_count != -1 else length
                extracted_values += extracted_values_collection[
                                    start_index: end_index]  # Getting only the desired subset of extracted data

            label_data_dict[a_field_name] = extracted_values

        return label_data_dict
