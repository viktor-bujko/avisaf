#!/usr/bin/env python3

import json
import os
import sys


def get_span_indexes(text, span):
    """

    :param text:
    :param span:
    :return:
    """
    result = {}
    positions = [pair for pair in get_all_indexes(text, span) if pair != (-1, -1)]
    if not positions:
        positions = [(-1, -1)]
    result[span] = positions
    return result


def get_all_indexes(text, span):
    """
    Method which returns the indexes of a substring of a string, if such substring
    exists.
    :param text: The sentence string.
    :param span:     A substring to be searched for.
    :return:         (start_index, end_index) tuple.
    """
    try:
        result = []
        start_index = str(text).index(span)
        end_index = start_index + len(span)
        result.append((start_index, end_index))
        others = [pair for pair in get_all_indexes(text[end_index:], span)]
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

    :param match_text:
    :param entities_dict:
    :return:
    """
    ent_list = entities_dict['entities']  # list of entities in the form of (start_index, end_index, label)
    for (start, end, label) in ent_list:
        print(match_text[start:end], label)


def get_training_data():
    """

    :return:
    """
    with open(os.path.expanduser('~/Documents/avisaf-ner/training_data_parts.json'), mode='r') as file:
        content = file.read()
        TR_DATA = json.loads(content)
        return TR_DATA


if __name__ == '__main__':
    phrase = sys.argv[1]

    print(get_spans_indexes(phrase, sys.argv[2:]))
