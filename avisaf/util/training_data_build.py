#!/usr/bin/env python3
"""
Training data build is a module responsible mainly for training data
manipulation. The module is used for sorting training annotations, removing
overlaps from entity annotations as well as file content formatting.
"""

import json
from pathlib import Path
from util.data_extractor import JsonDataExtractor


def fetch_and_sort_annotations(file_path: Path):
    """
    Function that sorts the (start_index, end_index, label) tuples based by
    their position in the given text. Sorting is done for every line of a JSON
    training data file.

    :type file_path:  Path
    :param file_path: Path of the JSON file containing the list of
                      (text, entities) tuples, which will have entities sorted.

    :return: Returns sorted annotation data.
    """

    file_path = file_path.resolve()
    extractor = JsonDataExtractor([file_path])
    # training_data = extractor.get_ner_training_data()
    sorted_training_data = []

    for text, annotation in extractor.get_ner_training_data():
        annot_list = annotation["entities"]  # get entities list from "entities" key in the annotation dictionary
        sorted_list = sorted(annot_list)  # , key=lambda tple: (tple[0], tple[1], tple[2])
        # sort entities
        sorted_training_data.append((text, {"entities": sorted_list}))  # recreate new, sorted dictionary

    with file_path.open(mode="w") as file:  # write the result to the same file
        json.dump(sorted_training_data, file)

    pretty_print_training_data(file_path)  # pretty print the file

    return sorted_training_data


def remove_overlaps(annotations_dict: dict) -> dict:
    """Removes overlapping annotations from the annotations_dict['entities'] list
    of (start_index, end_index, label) tuples.

    :type annotations_dict: dict
    :param annotations_dict: The dictionary containing the annotations list
        under 'entities' key.

    :return: The list of new annotations without overlaps.
    """
    for _ in range(2):

        # get entities list from "entities" key in the annotation dictionary
        entities_list = annotations_dict["entities"]
        if not entities_list:
            return {"entities": []}
        index = 0
        current_triplet = entities_list[index]
        keep_list = set()
        lookahead = 1

        while 0 <= index < len(entities_list) - 1:
            if (index + lookahead) >= len(entities_list):
                break

            next_triplet = entities_list[index + lookahead]
            if current_triplet == next_triplet:
                # possible duplicates don't matter when adding to set
                keep_list.add(current_triplet)
                lookahead += 1
                continue

            triplets_to_keep = decide_overlap_between(current_triplet, next_triplet)
            current_triplet = (*current_triplet,)  # remapping from [a, b, c] to tuple (a, b, c)
            if triplets_to_keep == [next_triplet] and current_triplet in keep_list:  # modifying a list to a tuple
                keep_list.remove(current_triplet)
                lookahead = 0

            current_triplet = triplets_to_keep[-1]
            index = entities_list.index(current_triplet)  # moving the index forward

            for to_keep in triplets_to_keep:
                # possible duplicates don't matter when adding to set
                keep_list.add((*to_keep,))
            # Only the first triplet is kept -> we skip the next one by increasing the lookahead
            # Otherwise; default lookahead of 1 is used
            lookahead = lookahead + 1 if triplets_to_keep == [current_triplet] else 1
        
        annotations_dict = {"entities": sorted(list(keep_list))}

    return annotations_dict


def decide_overlap_between(entity_triplet, next_triplet):
    """Detects whether there is an overlap between two triplets in the given text.
    If the two entities have the same label, shorter triplet is removed.

    :type entity_triplet: tuple
    :param entity_triplet: First  (start_index, end_index, label) entity descriptor.
    :type next_triplet: tuple
    :param next_triplet: Second (start_index, end_index, label) entity descriptor.

    :return: Returns the triplet to be removed - the one which represents a
        shorter part of the text.
    """
    entity_start, entity_end, entity_label = entity_triplet  # first entity description
    next_start, next_end, next_label = next_triplet  # second entity description

    x = set(range(entity_start, entity_end))
    y = range(next_start, next_end)

    # if an overlap is detected between two tuples.
    if not x.intersection(y):
        # keeping both entity triplets and skipping one next triplet (e.g. next_triplet)
        return [entity_triplet, next_triplet]

    # Entities overlap below
    # we keep longer of the triplets - usually the one which is "more" correct
    if (entity_end - entity_start) > (next_end - next_start):
        return [entity_triplet]
    elif (entity_end - entity_start) != (next_end - next_start):
        # entity is strictly shorter
        return [next_triplet]
    else:
        # both entities have equal lengths -> decide by rules
        rules = {
            ("ABBREVIATION", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "ABBREVIATION"): next_triplet,
            #
            ("AIRPLANE", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "AIRPLANE"): next_triplet,
            #
            ("CREW", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "CREW"): next_triplet,
            #
            ("WEATHER", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "WEATHER"): next_triplet,
            #
            ("AVIATION_TERM", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "AVIATION_TERM"): next_triplet,
            #
            ("AIRPORT_TERM", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "AIRPORT_TERM"): next_triplet,
            #
            ("FLIGHT_PHASE", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "FLIGHT_PHASE"): next_triplet,
            #
            ("ALTITUDE", "NAV_WAYPOINT"): entity_triplet,
            ("NAV_WAYPOINT", "ALTITUDE"): next_triplet,
            #
            ("AIRPLANE", "ABBREVIATION"): entity_triplet,
            ("ABBREVIATION", "AIRPLANE"): next_triplet
        }
        entity_to_keep = rules.get((entity_label, next_label))
        if entity_to_keep is None:
            return [next_triplet]
        else:
            return [entity_to_keep]


def pretty_print_training_data(file_path: Path):
    """Prints each tuple object of the document in a new line instead of a single
    very long line.

    :type file_path: Path
    :param file_path: The path of the file to be rewritten.
    """

    file_path = file_path.resolve()

    with file_path.open(mode="r") as file:
        content = json.load(file)

    with file_path.open(mode="w") as file:
        file.write("[")
        for i, entry in enumerate(content):
            json.dump(entry, file)
            if i != len(content) - 1:
                file.write(",\n")
            else:
                file.write("\n")
        file.write("]")


def write_sentences():
    """A loop which prompts a user to input a sentence which will be annotated
    later. The function ends when string 'None' is detected

    :return: The list of user-written sentences.
    """
    result = []
    sentence = input('Write a sentence or "None" to exit the loop: ')

    while sentence != "None":
        result.append(sentence)
        sentence = input("Write a sentence: ")

    return result
