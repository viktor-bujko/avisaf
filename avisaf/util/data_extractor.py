#!/usr/bin/env python3
"""The data extractor module is responsible for getting raw text data and
training data used by other modules.
"""

import pandas as pd
import json
import logging
from pathlib import Path
import numpy as np
from typing import Union


class DataExtractor:

    def __init__(self, file_paths: list):
        self.file_paths = file_paths

    def extract_data(self, a: list, lines_count: int = -1, start_index: int = 0) -> dict:
        pass


class JsonDataExtractor(DataExtractor):
    # TODO: This class should contain methods for working with json files
    # get_entities, get_training_data

    def __init__(self, file_paths: list):
        super().__init__(file_paths)

    def extract_data(self) -> dict:
        pass


def get_entities(entities_file_path: Union[str, Path] = None) -> dict:
    # Works with entities_labels.json file = only a simple json list
    # Probably will be moved to JsonDataExtractor
    """Function which reads given JSON file supposed to contain the list of user
    defined entity labels.

    :type entities_file_path: str, Path
    :param entities_file_path: The path to the JSON file containing the list
        of entity labels.

    :return: Returns the list of available entity labels.
    """
    if entities_file_path is None:
        # entities_file_path = Path(Path().resolve(), 'avisaf_ner/entities_labels.json')
        entities_file_path = find_file_by_path("entities_labels.json")
        if entities_file_path is None:
            raise FileNotFoundError()

    entities_file_path = Path(entities_file_path).resolve()

    with entities_file_path.open(mode="r") as entities_file:
        return json.load(entities_file)


def get_training_data(training_data_file_path: Path):
    # Works with (text, annotations) list JSON file
    # TODO: Probably will be moved to JsonDataExtractor
    """Function which reads given JSON file supposed to contain the training data.
    The training data are supposed to be a list of (text, annotations) tuples.

    :type training_data_file_path: Path
    :param training_data_file_path: The path to the JSON file containing the
        training data.

    :return: Returns the JSON list of (text, annotations) tuples.
    """
    if not training_data_file_path:
        msg = "Training data file path cannot be None"
        logging.error(msg)
        raise TypeError(msg)

    if not training_data_file_path.is_absolute():
        training_data_file_path = training_data_file_path.resolve()

    with training_data_file_path.open(mode="r") as tr_data_file:
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
        msg = "The file_path to have narratives extracted from cannot be None"
        logging.error(msg)
        raise TypeError(msg)

    file_path = str(file_path) if file_path.is_absolute() else str(file_path.resolve())

    report_df = pd.read_csv(file_path, skip_blank_lines=True, index_col=0, header=[0, 1])
    report_df.columns = report_df.columns.map("_".join)

    try:
        narratives1 = report_df["Report 1_Narrative"].values.tolist()
        calls1 = report_df["Report 1_Callback"].values.tolist()
        narratives2 = report_df["Report 2_Narrative"].values.tolist()
        calls2 = report_df["Report 2_Callback"].values.tolist()

    except KeyError:
        logging.error("No such key was found")
        return None

    length = len(narratives1)
    lists = [narratives1, calls1, narratives2, calls2]

    end_index = start_index + lines_count

    result = []
    for index in range(start_index, length):
        if lines_count != -1 and index >= end_index:
            break
        result.append(" ".join([str(lst[index]) for lst in lists if str(lst[index]) != "nan"]))

    return result


def find_file_by_path(file_path: Union[Path, str], max_iterations: int = 5):
    # Static utility function not tied to any object
    """

    :param max_iterations:
    :param file_path:
    :return:
    """

    current_dir = Path().resolve()

    for _ in range(max_iterations):  # Searching at most max_iterations times.
        if current_dir.parent is None or current_dir.parent == current_dir:
            break

        requested_file = current_dir.joinpath(Path(file_path)).resolve()

        if requested_file.exists():
            return requested_file

        current_dir = current_dir.parent

    return None


class CsvAsrsDataExtractor(DataExtractor):
    def __init__(self, file_paths: list):
        super().__init__(file_paths)

    def extract_data(self, field_names: list, lines_count: int = -1, start_index: int = 0) -> dict:
        """

        :param field_names:
        :param lines_count:
        :param start_index:
        :return:
        """

        skipped_files = 0

        label_data_dict = {}
        for field_name in field_names:
            extracted_values = []
            for file_path in self.file_paths:
                if not Path(file_path).exists():
                    skipped_files += 1
                    if skipped_files == len(self.file_paths):
                        raise ValueError(f"Any of the given files {self.file_paths} exists")
                    continue

                requested_file = find_file_by_path(file_path)
                if requested_file is None:
                    logging.error(f"The file given by \"{file_path}\" path was not found in the given range.")
                    # Ignoring the file with current file path
                    continue
                with open(requested_file) as file:
                    csv_dataframe = pd.read_csv(file, skip_blank_lines=True, header=[0, 1])
                logging.debug(f"File {requested_file} is closed: {file.closed}")
                csv_dataframe.columns = csv_dataframe.columns.map("_".join)
                csv_dataframe = csv_dataframe.replace(np.nan, "", regex=True)

                try:
                    extracted_values_collection = csv_dataframe[field_name].values.tolist()
                except KeyError:
                    logging.error(f"\"{field_name}\" is not a correct field name. Please make sure the column name is in format \"FirstLineTitle_SecondLineTitle\"")
                    continue

                length = len(extracted_values_collection)
                end_index = start_index + lines_count if lines_count != -1 else length
                extracted_values += extracted_values_collection[start_index:end_index]  # Getting only the desired subset of extracted data

            label_data_dict[field_name] = np.array(extracted_values)

        return label_data_dict
