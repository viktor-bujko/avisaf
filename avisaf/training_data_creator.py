#!/usr/bin/env python3

from spacy.matcher import PhraseMatcher
from util.file_choice import extract_narratives
from util.indexing import get_spans_indexes
import spacy
import json
import os
import sys

PATH = '~/Documents/avisaf_ner/generated_annotation_data.json'


def create_tr_data():
    TRAINING_DATA = []

    # create first rule-based keys corresponding to acft parts
    with open(os.path.expanduser('~/Documents/avisaf_ner/acft_parts_list.json'), mode='r') as keys_file:
        # replace fixed file path with path argument for dictionary
        KEYS = json.loads(keys_file.read())  # patterns to be matched

    # get testing texts;
    TEXTS = list(extract_narratives())

    # create NLP analyzer object
    nlp = spacy.load('en_core_web_sm')
    # create PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)

    # process the keys and store their values in the patterns list
    patterns = list(nlp.pipe(KEYS))
    # add all patterns to the matcher
    matcher.add('ACFT_PART', None, *patterns)

    for doc in nlp.pipe(TEXTS[1101:]):
        matched_spans = [doc[start:end] for _, start, end in matcher(doc)]
        entities = [(span.start_char, span.end_char, 'ACFT_PART') for span in matched_spans]
        tr_example = (doc.text, {"entities": entities})
        TRAINING_DATA.append(tr_example)

    print(*TRAINING_DATA, sep='\n')


def write_sentences():
    phrase = input('Write a sentence: ')
    result = []
    while phrase != 'None':
        result.append(phrase)
        phrase = input('Write a sentence: ')

    return result


def annotate_texts(labels, file_path, start_index=0):
    n = input('How many texts should be annotated?: ')
    lines = int(n) if n.isdigit() else -1

    question = input('Save the annotated data in the file?: ')

    if file_path is not None:
        TEXTS = list(extract_narratives(lines=lines + start_index, file_path=file_path, start_index=start_index))
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
            with open(os.path.expanduser(PATH), mode='r') as file:
                old_content = json.loads(file.read())

            with open(os.path.expanduser(PATH), mode='w') as file:
                old_content.append(new_entry)
                json.dump(old_content, file)

    if question.startswith('y'):
        pretty_print_training_data(PATH)

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


if __name__ == '__main__':

    path = sys.argv[1]
    first_text_idx = int(sys.argv[2])
    ents = ["AIRPLANE", "CREW", "AIRPORT_TERM", "FLIGHT_PHASE", "AVIATION_TERM", "NAV_WAYPOINT", "ALTITUDE", "WEATHER", "ABBREVIATION"]
    print(annotate_texts(labels=ents, file_path=path, start_index=first_text_idx))
    pretty_print_training_data(PATH)
