#!/usr/bin/env python3

import json
import os


def get_span_indexes(text, span):
    """
    Launches the search for a given span in given text. Allows also looking for more complex,
    token-composed spans.
    :param text:
    :param span:
    :return:
    """

    result = {}
    positions = [pair for pair in find_indexes(text, span, start_offset=0) if pair != (-1, -1)]
    if positions:               # if positions list contains something else apart (-1, -1) tuple -> a match was found
        result[span] = positions
    else:
        result[span] = []
    return result


def find_indexes(text, span, start_offset):
    """
    Method which returns the indexes of a substring of a string, if such substring
    exists, otherwise return (-1, -1) tuple.
    :param text:         The sentence string.
    :param span:         A substring to be searched for.
    :param start_offset: TO BE DONE
    :return:             (start_index, end_index) tuple or (-1, -1) if span is not in the text.
    """
    try:
        result = []
        start_index = str(text).index(span, start_offset)   # find starting index of the span in the text
        end_index = start_index + len(span)                 # get the end_index of the span
        if not text[end_index].isalnum():               # ensure, that the span is not the substring of another word
            result.append((start_index, end_index))
            others = [pair for pair in find_indexes(text, span, end_index)]
            result += others
        return result

    except ValueError:
        return [(-1, -1)]


def get_spans_indexes(sentence, spans):
    """
    Same as get_span_indexes function, but takes a list of spans instead of a single span.
    :param sentence: The sentence string.
    :param spans:    List of substrings to be searched for.
    :return:         List of (start_index, end_index) pairs.
    """
    RESULT = []
    for span in spans:
        res = get_span_indexes(sentence, span)
        RESULT.append(res)
    return RESULT


def print_matches(match_text, entities_dict):
    """
    Function which takes a text, and its annotated entities,
    and prints the annotated text along with its labeled entity.
    :param match_text:      The text to be searched in for labeled entities.
    :param entities_dict:   The dictionary with 'entities key containing the list of
                            (start_index, end_index, label) entities of the text.
    """
    ent_list = entities_dict['entities']  # list of entities in the form of (start_index, end_index, label)
    for (start, end, label) in ent_list:
        print(match_text[start:end], label)


def get_training_data(path):
    """
    Gets the training data from a given file.
    :param path: The file path to the training data JSON file.
    :return: The list of (text, annotations) tuples.
    """
    with open(path, mode='r') as tr_data_file:
        TR_DATA = json.loads(tr_data_file.read())
        return TR_DATA


if __name__ == '__main__':
    # phrase = sys.argv[1]

    with open(os.path.expanduser('~/Documents/avisaf_ner/data_files/auto_annotated_data.json'), mode='r') as file:
        json_list = json.load(file)

    for text, entities in json_list:
        print_matches(text, entities)
