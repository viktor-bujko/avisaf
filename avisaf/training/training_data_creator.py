#!/usr/bin/env python3
"""Training data creator is the module responsible for data annotation which can
be done either manually or automatically. Automatic data annotation is done
by using pre-created set of Matcher or PhraseMatcher rules which are then
applied to each of the given texts. Manual annotation is done by showing
the texts to the user and then letting him choose the words to be annotated
as well as the entity labels used for the chosen phrases.
"""

import sys
import os
import json
import spacy
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher
# importing own modules used in this module
from avisaf.util.indexing import get_spans_indexes, entity_trimmer
import avisaf.util.training_data_build as train
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# looking for the project root
path = Path(__file__)
while not str(path.resolve()).endswith('avisaf'):
    path = path.parent.resolve()

SOURCES_ROOT_PATH = Path(path).resolve()
if str(SOURCES_ROOT_PATH) not in sys.path:
    sys.path.append(str(SOURCES_ROOT_PATH))


def annotate_auto(patterns_file_path: Path, label_text: str,
                  model='en_core_web_md', tr_src_file: Path = None,
                  extract_texts: bool = False, use_phrasematcher: bool = False,
                  save: bool = False, verbose: bool = False):
    """Automatic annotation tool. The function takes a file which has to contain a
    JSON list of rules to be matched. The rules are in the format compatible
    with spaCy Matcher or PhraseMatcher objects. Rule recognition is done by
    spaCy pattern matching in the given text.
    
    :type patterns_file_path: Path
    :param patterns_file_path: String representing a path to the file with
        words to be matched (glossary etc).
    :type label_text: str
    :param label_text: The text of the label of an entity.
    :type model: str, Path
    :param model: Model to be loaded to spaCy. Either a valid spaCy pre-trained
        model or a path to a local model.
    :type tr_src_file: Path
    :param tr_src_file: Training data source file path. JSON file is supposed to
        contain list of (text, annotations) tuples, where the text is the string
        and annotations represents a dictionary with list of (start, end, label)
        entity descriptors.
    :type extract_texts: bool
    :param extract_texts: A flag indicating whether new texts should be searched
        for.
    :type use_phrasematcher: bool
    :param use_phrasematcher: A flag indicating whether Matcher or PhraseMatcher
        spaCy object is used.
    :type save: bool
    :param save: A flag indicating whether the data should be saved in the same
        tr_src_file.
    :type verbose: bool
    :param verbose: A flag indicating verbose stdout printing.
    """

    from avisaf.util.data_extractor import get_narratives
    # almost result list - list containing all entities - including the overlaps
    tr_data_overlaps = []
    tr_src_file = tr_src_file.resolve()
    patterns_file_path = patterns_file_path.resolve()

    if extract_texts or tr_src_file is None:
        # get testing texts
        texts = list(get_narratives())  # file_path is None
        entities = None
    else:
        with tr_src_file.open(mode='r') as tr_data_file:
            # load the file containing the list of training ('text string', entity dict) tuples
            tr_data = json.load(tr_data_file)
            texts = [text for text, _ in tr_data]
            entities = [ents for _, ents in tr_data]

    # create NLP analyzer object of the model
    nlp = spacy.load(model)
    with patterns_file_path.open(mode='r') as pttrns_file:
        patterns = json.load(pttrns_file)  # phrase/patterns to be matched

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

    print(f'Using {matcher}', flush=verbose)

    for doc in nlp.pipe(texts, batch_size=100):
        matches = matcher(doc)
        matched_spans = [doc[start:end] for match_id, start, end in matches]
        print(f'Doc index: {texts.index(doc.text)}', f'Matched spans: {matched_spans}', flush=verbose)
        new_entities = [(span.start_char, span.end_char, label_text) for span in matched_spans]
        # following line of code also resolves situation when the entities dictionary is None
        tr_example = (doc.text, {"entities": new_entities})
        if entities is not None:
            doc_index = texts.index(doc.text)
            old_entities = list(entities[doc_index]["entities"])
            new_entities = new_entities + old_entities
            tr_example = (doc.text, {"entities": new_entities})

        tr_data_overlaps.append(tr_example)

    training_data = []  # list will contain training data without overlaps

    for text, annotations in tr_data_overlaps:
        new_annotations = train.remove_overlaps_from_dict(annotations)
        training_data.append((text, {"entities": new_annotations}))

    if save and tr_src_file is not None:
        with tr_src_file.open(mode='w') as file:
            json.dump(training_data, file)

        train.remove_overlaps_from_file(tr_src_file)
        entity_trimmer(tr_src_file)
        train.pretty_print_training_data(tr_src_file)
    else:
        print(*training_data, sep='\n')

    return training_data


def annotate_man(file_path: Path, lines: int = -1,
                 labels_path: Path = None, start_index: int = 0,
                 save: bool = True):
    """
    Manual text annotation tool. A set of texts from file_path parameter
    starting with start_index is progressively printed in order to be annotated
    by labels given in the labels_path.

    :type labels_path:  Path
    :param labels_path: Path to the file containing available entity labels,
        defaults to None.
    :type file_path:    Path
    :param file_path:   The path to the file containing texts to be annotated.
                        If None, then a user can write own sentences and
                        annotate them.
    :type lines:        int
    :param lines:       The number of texts to be annotated (1 text = 1 line),
        defaults to -1 - means all the lines.
    :type start_index:  int
    :param start_index: The index of the first text to be annotated.
    :type save:         bool
    :param save:        A flag indicating whether the result of the annotation
                        should be saved.

    :return:            List of texts and its annotations.
    """
    from avisaf.util.data_extractor import get_entities, get_narratives

    labels = get_entities(labels_path) if labels_path is not None else get_entities()

    if file_path is not None:
        if file_path.exists():
            if file_path.suffix == '.csv':
                texts = get_narratives(
                    lines_count=lines,
                    file_path=file_path,
                    start_index=start_index
                )
            else:
                with file_path.open(mode='r') as file:
                    texts = json.load(file)
        else:
            # use given argument as the text to be annotated
            texts = [str(file_path)]
            print()  # print an empty line

    else:
        texts = train.write_sentences()

    result = []

    # if we don't want to annotate all texts
    if lines != -1:
        texts = texts[start_index:start_index + lines]

    for text in texts:
        ent_labels = []
        print(text)
        print()  # print an empty line
        words = input('Write all words you want to annotate (separated by a comma): ')
        spans = set([word.strip() for word in words.split(',') if word.strip()])

        if not spans:
            new_entry = (text, {"entities": []})
            result.append(new_entry)
        else:
            # find positions of "spans" string list items in the text
            found_occurs = get_spans_indexes(text, list(spans))
            for occur_dict in found_occurs:
                key = list(occur_dict.keys())[0]  # only the first key is desired
                matches = occur_dict[key]
                label = input(f"Label '{key}' with an item from: {list(enumerate(labels))} or type 'NONE' to skip: ")\
                    .upper()
                if label not in labels and not label.isdigit():  # when there is no suitable label in the list
                    continue
                if label.isdigit():
                    ent_labels += [(start, end, labels[int(label)]) for start, end in matches]  # create the tuple
                else:
                    # same as above, but entity label text is directly taken
                    ent_labels += [(start, end, label) for start, end in matches]

            ents_no_overlaps = train.remove_overlaps_from_dict({"entities": ent_labels})

            new_entry = (text, {"entities": ents_no_overlaps})
            result.append(new_entry)
        print()  # print an empty line

        if save:
            man_training_data_file = Path('data_files', 'training_data', 'man_annotated_data.json').resolve()
            man_training_data_file.touch(exist_ok=True)

            # if the file is not empty
            if len(man_training_data_file.read_bytes()) != 0:
                # rewrite the current content of the file
                with open(os.path.expanduser(man_training_data_file), mode='r') as file:
                    old_content = json.load(file)
            else:
                old_content = []

            with open(os.path.expanduser(man_training_data_file), mode='w') as file:
                old_content.append(new_entry)
                json.dump(old_content, file)
                print(f"Content in the {man_training_data_file.relative_to(SOURCES_ROOT_PATH.parent)} updated.\n")

            train.pretty_print_training_data(man_training_data_file)

    return result


def build_feature_matrices_from_texts(texts: list, target_labels: list, text_vectorizer=TfidfVectorizer, target_label_filter: list = None):
    """

    :param target_label_filter:
    :param texts:
    :param target_labels:
    :param text_vectorizer:
    :return:
    """

    texts, target_labels, encoding = get_unique_string_labels(texts, target_labels, target_label_filter)

    if texts.shape != target_labels.shape:
        raise ValueError('The number of training examples is not equal to the the number of labels.')

    # stops = spacy.load('en_core_web_md').Defaults.stop_words

    texts_vectors = text_vectorizer(
        stop_words='english',
        lowercase=False,
        max_features=10000
    ).fit_transform(texts)  # .toarray() -> create a matrix from csr_matrix

    return texts_vectors, np.array(target_labels), encoding


def get_unique_string_labels(texts: list, target_labels: list, target_label_filter: list = None):
    """

    :param target_label_filter:
    :param texts:
    :param target_labels:
    :return:
    """

    new_texts, new_labels = [], []
    unique_labels = set()

    for text_idx, text in enumerate(texts):
        labels = target_labels[text_idx].split(';')
        for label in labels:
            label = label.strip()
            if target_label_filter is not None and label not in target_label_filter:
                continue
            new_texts.append(text)
            new_labels.append(label)
            unique_labels.add(label)

    if target_label_filter is not None:
        for label in target_label_filter:
            if label not in unique_labels:
                print(f'The label has not been found. Check whether "{label}" is correct category spelling.', file=sys.stderr)

    unique_labels = sorted(unique_labels)
    new_labels, encoding = encode_labels(unique_labels, new_labels)
    _, counts = np.unique(new_labels, return_counts=True)
    print(dict(zip(unique_labels, counts)))

    return np.array(new_texts), np.array(new_labels), encoding


def encode_labels(unique: list, labels: list):
    """

    :param unique:
    :param labels:
    :return:
    """
    encoded_labels = []

    for label in labels:
        encoded_labels.append(unique.index(label))

    encoding = dict(zip(range(len(unique)), unique))

    return encoded_labels, encoding


def normalize_data_distribution(data, target, deviation_percentage: float = 0.05):
    """

    :param deviation_percentage:
    :param data:
    :param target:
    :return:
    """

    distribution_counts, dist = get_data_distribution(target)

    most_even_distribution = 1 / len(distribution_counts)

    filtered_indices = set()
    more_present_idxs = np.where(dist > most_even_distribution)[0]
    # distribution_surplus = dist[more_present_index] - most_even_distribution
    # examples_to_remove = int(distribution_surplus * data.shape[0])
    examples_to_remove = int(np.sum(
        (distribution_counts[more_present_idxs] - np.min(distribution_counts)) * np.random.uniform(
            low=1 - deviation_percentage,
            high=1 + deviation_percentage
        )
    ))

    repeated_match = 0
    while examples_to_remove > 0:
        rnd_index = np.random.randint(0, data.shape[0])
        should_be_filtered = target[rnd_index] in more_present_idxs
        if not should_be_filtered:
            continue

        # rnd_index should be filtered here

        if rnd_index in filtered_indices:
            repeated_match += 1
        else:
            filtered_indices.add(rnd_index)
            examples_to_remove -= 1
            repeated_match = 0

        # Avoiding "infinite loop" caused by always matching already filtered examples
        # Giving up on filtering the exact number of examples -> The distribution will be less even
        examples_to_remove = examples_to_remove - 1 if (repeated_match > 0 and repeated_match % 100 == 0) else examples_to_remove

    arr_filter = [idx not in filtered_indices for idx in range(data.shape[0])]

    return data[arr_filter], target[arr_filter]


def get_data_distribution(target: list):
    """

    :param target:
    :return:
    """

    distribution_counts = np.histogram(target, bins=len(np.unique(target)))[0]
    dist = distribution_counts / np.sum(distribution_counts)  # Gets percentage presence of each class in the data

    return distribution_counts, dist
