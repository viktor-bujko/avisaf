#!/usr/bin/env python3

from spacy.matcher import PhraseMatcher
from util.data_extractor import get_narratives, get_entities
from util.indexing import get_spans_indexes
import spacy
import json
import os
import sys

TRAINING_DATA_FILE_PATH = '~/Documents/avisaf_ner/generated_annotation_data.json'


def create_matcher_tr_data(kw_file_path, label_text, model='en_core_web_md', tr_src_file=None, extract_texts=False):
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
    :return:
    """
    # result list
    TRAINING_DATA = []

    with open(kw_file_path, mode='r') as keys_file:
        # replace fixed file path with path argument for dictionary
        KEYWORDS = json.loads(keys_file.read())  # phrase-patterns to be matched

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
    # create PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)

    # process the keys and store their values in the patterns list
    patterns = list(nlp.pipe(KEYWORDS))
    # add all patterns to the matcher
    matcher.add(label_text, None, *patterns)

    for doc in nlp.pipe(TEXTS, batch_size=50):
        print(TEXTS.index(doc.text))
        matched_spans = [doc[start:end] for match_id, start, end in matcher(doc)]
        entities = [(span.start_char, span.end_char, label_text) for span in matched_spans]
        tr_example = (doc.text, {"entities": entities})
        if ENTITIES is not None:
            doc_index = TEXTS.index(doc.text)
            exist_ent_list = list(ENTITIES[doc_index]["entities"])
            entities = entities + exist_ent_list
            tr_example = (doc.text, {"entities": entities})

        TRAINING_DATA.append(tr_example)

    # print(*TRAINING_DATA, sep='\n')

    if input("Save file?: ").startswith("y") and tr_src_file is not None:
        with open(tr_src_file, mode='w') as file:
            json.dump(TRAINING_DATA, file)

        pretty_print_training_data(tr_src_file)

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
            with open(os.path.expanduser(TRAINING_DATA_FILE_PATH), mode='r') as file:
                old_content = json.loads(file.read())

            with open(os.path.expanduser(TRAINING_DATA_FILE_PATH), mode='w') as file:
                old_content.append(new_entry)
                json.dump(old_content, file)

    if question.startswith('y'):
        pretty_print_training_data(TRAINING_DATA_FILE_PATH)

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

    """path = sys.argv[1]
    first_text_idx = int(sys.argv[2])
    ents = get_entities()
    print(annotate_texts(labels=ents, file_path=path, start_index=first_text_idx))
    pretty_print_training_data(TRAINING_DATA_FILE_PATH)"""

    create_matcher_tr_data("/home/viktor/Documents/avisaf_ner/resources/crew_list.json",
                           "CREW",
                           tr_src_file="/home/viktor/Documents/avisaf_ner/resources/training_data_parts.json")
