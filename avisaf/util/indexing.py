#!/usr/bin/env python3
"""Indexing module is used for searching for the spans in the texts.
"""

import json
import re
from pathlib import Path


def get_span_indexes(text: str, span: str):
    """Launches the search for a given span in given text. Allows also looking
    for more complex, token-composed spans.

    :type text: str
    :param text: The source text to be searched in for the span.
    :type span: str
    :param span: The substring which is searched in the text.

    :return: The dictionary which contains all occurrences of the searched span
        in the text.
    """

    result = {}
    positions = [pair for pair in find_indexes(text, span, start_offset=0) if pair != (-1, -1)]
    if positions:  # if positions list contains something else apart (-1, -1) tuple -> a match was found
        result[span] = positions
    else:
        result[span] = []
    return result


def find_indexes(text: str, span: str, start_offset: int):
    """Method which returns the indexes of a substring of a string, if such
    substring exists, otherwise return (-1, -1) tuple.

    :type text: str
    :param text: The source text to be searched in for the span.
    :type span: str
    :param span: A substring to be searched for in the text.
    :type start_offset: int
    :param start_offset: The index of the text where the search is started.

    :return: (start_index, end_index) tuple or (-1, -1) if span is not in the text.
    """
    try:
        result = []
        start_index = str(text).index(
            span, start_offset
        )  # find starting index of the span in the text
        end_index = start_index + len(span)  # get the end_index of the span
        if not text[
            end_index
        ].isalnum():  # ensure, that the span is not the substring of another word
            result.append((start_index, end_index))
            others = [pair for pair in find_indexes(text, span, end_index)]
            result += others
        return result

    except ValueError:
        return [(-1, -1)]


def get_spans_indexes(sentence: str, spans: list):
    """Same as get_span_indexes function, but takes a list of spans instead of
    a single span.

    :type sentence: str
    :param sentence: The sentence string.
    :type spans: list
    :param spans: List of substrings to be searched for.

    :return: List of (start_index, end_index) pairs.
    """
    result = []
    for span in spans:
        res = get_span_indexes(sentence, span)
        result.append(res)
    return result


# TODO: To remove
def print_matches(match_text: str, entities_dict: dict):
    """Function which takes a text, its annotated entities and prints the
    annotated text along with its labeled entity.

    :type match_text: str
    :param match_text: The text to be searched in for labeled entities.
    :type entities_dict: dict
    :param entities_dict: The dictionary with 'entities key containing the list
        of (start_index, end_index, label) entities of the text.
    """
    ent_list = entities_dict["entities"]  # list of entities in the form of (start_index, end_index, label)
    for (start, end, label) in ent_list:
        print(f"'{match_text[start:end]}'", f'"{label}"')


def get_training_data(path: Path):
    """Gets the training data from a given file.

    :type path: Path
    :param path: The file path to the training data JSON file.

    :return: The list of (text, annotations) tuples.
    """
    path = path if path.is_absolute() else path.resolve()

    with path.open(mode="r") as tr_data_file:
        tr_data = json.loads(tr_data_file.read())
        return tr_data


def entity_trimmer(data_file_path: Path):
    """Function responsible for removing leading and trailing white spaces from
    entity spans.

    :type data_file_path: Path
    :param data_file_path: Data in spaCy JSON format to have leading and trailing
        whitespaces removed.

    :return: Returns the list without leading/trailing whitespaces.
    """
    invalid_span_tokens = re.compile(r"\s")
    clean_data = []

    with data_file_path.open(mode="r") as data_file:
        data = json.load(data_file)

    for text, annotations in data:
        entities = annotations["entities"]  # get annotations list from dictionary
        correct_entities = []
        for ent_start, ent_end, ent_label in entities:
            correct_start = ent_start
            correct_end = ent_end
            while correct_start < len(text) and invalid_span_tokens.match(
                text[correct_start]
            ):
                correct_start += (
                    1  # if a leading whitespace is detected, start position increases
                )
            while correct_end > 1 and invalid_span_tokens.match(text[correct_end - 1]):
                correct_end -= (
                    1  # if a trailing whitespace is detected, end position decreases
                )
            correct_entities.append([correct_start, correct_end, ent_label])

        clean_data.append([text, {"entities": correct_entities}])

    with data_file_path.open(mode="w") as data_file:
        json.dump(clean_data, data_file)

    return clean_data


"""
if __name__ == "__main__":
    path_str = sys.argv[1]

    with open(os.path.expanduser(path_str), mode="r") as file:
        json_list = json.load(file)

    for text_str, entities_out in json_list:
        print_matches(text_str, entities_out)
"""