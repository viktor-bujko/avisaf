#!/usr/bin/env python3

import sys
sys.path.append('/home/viktor/Documents/avisaf_ner/avisaf')

from spacy.matcher import PhraseMatcher, Matcher
from util.data_extractor import get_narratives, get_entities, get_training_data
from util.indexing import get_spans_indexes
import spacy
import json
import os
from pathlib import Path

MAN_TRAINING_DATA_FILE_PATH = '../../data_files/man_annotated_data.json'


def create_matcher_tr_data(kw_file_path,
                           label_text,
                           model='en_core_web_md',
                           tr_src_file=None,
                           extract_texts=False,
                           use_phrasematcher=True):
    """
    Semi-automatic annotation tool. The function takes a file which should contain a list of keywords to be matched
    :type kw_file_path:  str
    :param kw_file_path: String representing a path to the file with words to be matched (glossary etc).
    :type label_text:    str
    :param label_text:   The text of the label of an entity.
    :type model:         str
    :param model:        Optional: Model to be loaded to spaCy.
    :type tr_src_file:   str
    :param tr_src_file:  Training data source file path.
    :type extract_texts: bool
    :param extract_texts: Flag indicating whether new texts should be searched for. Default False.
    :type use_phrasematcher:   bool
    :param use_phrasematcher:  Flag indicating whether Matcher or PhraseMatcher spaCy object is used.
                               Default is PhraseMatcher.
    :return:
    """
    # result list
    TRAINING_DATA = []

    if extract_texts or tr_src_file is None:
        # get testing texts
        TEXTS = list(get_narratives())  # file_path is None
        ENTITIES = None
    else:
        with open(tr_src_file, mode='r') as tr_data_file:
            # load the file containing the list of training ('text string', entity dict) tuples
            tr_data = json.load(tr_data_file)
            TEXTS = [text for text, _ in tr_data]
            ENTITIES = [entities for _, entities in tr_data]

    # create NLP analyzer object of the model
    nlp = spacy.load(model)
    with open(kw_file_path, mode='r') as keys_file:
        patterns = json.load(keys_file)  # phrase/patterns to be matched

    if use_phrasematcher:
        # create PhraseMatcher object
        matcher = PhraseMatcher(nlp.vocab, validate=True)
        # process the keys and store their values in the patterns list
        keywords = list(nlp.pipe(patterns))
        # add all patterns to the matcher
        matcher.add(label_text, keywords)
    else:
        # create Matcher object
        matcher = Matcher(nlp.vocab, validate=True)
        matcher.add(label_text, patterns)

    print(f'Using {matcher}')
    print(*patterns, sep='\n')
    matcher1 = Matcher(nlp.vocab)
    # matcher1.add("LETISKO",
    #             [[{"LOWER": {"IN": ["runway", "rwy"]}, "OP": "?"}, {"TEXT": {"REGEX": "(0[1-9]|[1-2][0-9]|3[0-6])(L|C|R)?"}}]])

    print('----------------------------------------------------------------------')
    for doc in nlp.pipe(TEXTS, batch_size=50):
        matches = matcher(doc) + matcher1(doc)
        matched_spans = [doc[start:end] for match_id, start, end in matches]
        print(f'Doc index: {TEXTS.index(doc.text)}', f'Matched spans: {matched_spans}')
        new_entities = [(span.start_char, span.end_char, label_text) for span in matched_spans]
        tr_example = (doc.text, {"entities": new_entities})
        if ENTITIES is not None:
            doc_index = TEXTS.index(doc.text)
            old_entities = list(ENTITIES[doc_index]["entities"])
            # equivalent for list(set(new_entities + old_entities))
            new_entities = new_entities + old_entities
            tr_example = (doc.text, {"entities": new_entities})

        TRAINING_DATA.append(tr_example)

    if input("Save file?: ").startswith("y") and tr_src_file is not None:
        with open(tr_src_file, mode='w') as file:
            json.dump(TRAINING_DATA, file)

        pretty_print_training_data(tr_src_file)
    else:
        print(*TRAINING_DATA, sep='\n')

    return TRAINING_DATA


def write_sentences():
    phrase = input('Write a sentence: ')
    result = []
    while phrase != 'None':
        result.append(phrase)
        phrase = input('Write a sentence: ')

    return result


def annotate_texts(labels, file_path, start_index=0):
    """
    Semi-automatic text annotation.
    :param labels:    Available entity labels.
    :param file_path: The path to the file containing texts to be annotated.
    :param start_index: The index of the first text to be taken from the file.
    :return:
    """
    n = input('How many texts should be annotated?: ')
    lines = int(n) if n.isdigit() else -1

    question = input('Save the annotated data in the file? (y/N): ')

    if file_path is not None:
        TEXTS = list(get_narratives(lines=lines, file_path=file_path, start_index=start_index))
    else:
        TEXTS = write_sentences()
    result = []

    for text in TEXTS:
        ent_labels = []
        print(text)
        words = input('\nWrite all words you want to annotate (separated by a comma): ')
        spans = set([word.strip() for word in words.split(',') if word.strip()])

        if not spans:
            new_entry = (text, {"entities": []})
            result.append(new_entry)
        else:
            found_occurs = get_spans_indexes(text, spans)
            for occur_dict in found_occurs:
                key = list(occur_dict.keys())[0]            # we want only the first key
                matches = occur_dict[key]
                label = input(f'Label \'{key}\' with one of the following {list(enumerate(labels))}: ').upper()
                if label == 'NONE':
                    continue
                if label.isdigit():
                    ent_labels += [(start, end, labels[int(label)]) for start, end in matches]
                else:
                    ent_labels += [(start, end, label) for start, end in matches]
            new_entry = (text, {"entities": ent_labels})
            result.append(new_entry)
        print('\n')
        if question.startswith('y'):
            with open(os.path.expanduser(MAN_TRAINING_DATA_FILE_PATH), mode='r') as file:
                old_content = json.loads(file.read())

            with open(os.path.expanduser(MAN_TRAINING_DATA_FILE_PATH), mode='w') as file:
                old_content.append(new_entry)
                json.dump(old_content, file)

    if question.startswith('y'):
        pretty_print_training_data(MAN_TRAINING_DATA_FILE_PATH)

    return result


def pretty_print_training_data(path):
    """
    Prints each tuple of the document in a new line.
    :param path: The path of the training data file.
    :return:
    """
    with open(os.path.expanduser(path), mode='r') as file:
        content = json.loads(file.read())

    with open(os.path.expanduser(path), mode='w') as file:
        file.write('[')
        for i, entry in enumerate(content):
            json.dump(entry, file)
            if i != len(content) - 1:
                file.write(',\n')
            else:
                file.write('\n')
        file.write(']')


def decide_overlap(entity_triplet, other_triplet, text):
    entity_start = entity_triplet[0]
    entity_end = entity_triplet[1]
    other_start = other_triplet[0]
    other_end = other_triplet[1]
    x = set(range(entity_start, entity_end))
    y = range(other_start, other_end)

    entity_text = text[entity_triplet[0]:entity_triplet[1]]
    other_text = text[other_triplet[0]:other_triplet[1]]

    if x.intersection(y):
        if len(entity_text) >= len(other_text):
            return other_triplet
        else:
            return entity_triplet

            # print(f'Entity: {entity_triplet}; Text: {entity_text}')
            # print(f'Entity: {other_triplet}; Text: {other_text}')
            # to_remove = input("Enter entity which you want to remove: ")
            # if to_remove == "1":
            #    return entity_triplet
            # if to_remove == "2":
            #    return other_triplet
    else:
        return None


def decide_overlaps(file_path):

    sort_annotations(file_path)
    train_data = get_training_data(file_path)
    result = []

    for text, annotations in train_data:
        entities_list = annotations['entities']
        to_rem = []
        for entity_triplet in entities_list:
            index = list(entities_list).index(entity_triplet)
            if index < len(entities_list) - 1:
                other_triplet = entities_list[index + 1]
                triplet_to_remove = decide_overlap(entity_triplet, other_triplet, text)
                if triplet_to_remove is not None:
                    # print(f'Removing: {triplet_to_remove}\n')
                    to_rem.append(triplet_to_remove)
        new_annot = [entity for entity in entities_list if entity not in to_rem]
        # print(f"NEW ANNOT: {new_annot}")
        result.append((text, {"entities": new_annot}))

    with open(file_path, mode='w') as file:
        json.dump(result, file)

    pretty_print_training_data(file_path)

    return result


def sort_annotations(file_path):

    tr_data = get_training_data(file_path)

    sorted_tr_data = []
    for text, annotation in tr_data:
        annot_list = annotation["entities"]
        sorted_list = sorted(annot_list, key=lambda tple: tple[0])
        sorted_tr_data.append((text, {"entities": sorted_list}))

    with open(file_path, mode='w') as file:
        json.dump(sorted_tr_data, file)

    pretty_print_training_data(file_path)


if __name__ == '__main__':

    """path_arg = sys.argv[1]
    first_text_idx = int(sys.argv[2])
    ents = get_entities()
    print(annotate_texts(labels=ents, file_path=path_arg, start_index=first_text_idx))
    pretty_print_training_data(TRAINING_DATA_FILE_PATH)"""

    create_matcher_tr_data("/home/viktor/Documents/avisaf_ner/data_files/altitude_list.json",
                           "ALTITUDE",
                           tr_src_file="/home/viktor/Documents/avisaf_ner/data_files/auto_annotated_data.json",
                           use_phrasematcher=False)

    decide_overlaps('/home/viktor/Documents/avisaf_ner/data_files/auto_annotated_data.json')
