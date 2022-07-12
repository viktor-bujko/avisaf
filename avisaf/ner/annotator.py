#!/usr/bin/env python3
"""Training data creator is the module responsible for data annotation which can
be done either manually or automatically. Automatic data annotation is done
by using pre-created set of Matcher or PhraseMatcher rules which are then
applied to each of the given texts. Manual annotation is done by showing
the texts to the user and then letting him choose the words to be annotated
as well as the entity labels used for the chosen phrases.
"""

import os
import json
import spacy
import logging
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher

# importing own modules used in this module
from avisaf.util.indexing import get_spans_indexes, entity_trimmer
import avisaf.util.training_data_build as train
from avisaf.util.data_extractor import get_entities, CsvAsrsDataExtractor

logger = logging.getLogger("avisaf_logger")


def get_current_texts_and_ents(train_data_file: Path, extract_texts: bool):
    if extract_texts:
        # get testing texts
        extractor = CsvAsrsDataExtractor([train_data_file])
        texts = list(extractor.get_narratives())
        entities = None

        return texts, entities

    with train_data_file.open(mode="r") as tr_data_file:
        # load the JSON file containing the list of training ('text string', entity dict) tuples
        texts, entities = zip(*json.load(tr_data_file))  # replacement for the code below

        # texts, entities = [], []
        # for text, ents in json.load(tr_data_file):
        #    texts.append(text)
        #    entities.append(ents)

    return texts, entities


def auto_annotation_handler(
        patterns_file_path: str,
        label_text: str,
        training_src_file: str,
        model="en_core_web_md",
        extract_texts: bool = False,
        use_phrasematcher: bool = False,
        save: bool = False,
        save_to: str = None,
):
    """

    :param patterns_file_path:
    :param label_text:
    :param training_src_file:
    :param model:
    :param extract_texts:
    :param use_phrasematcher:
    :type save: bool
    :param save: A flag indicating whether the data should be saved in the same
        tr_src_file.
    :param save_to:
    :return: List containing automatically annotated generated training data.
    """

    if training_src_file is None:
        logger.error("The training data src file path cannot be None")

    if patterns_file_path is None:
        logger.error("File with patterns supposed to be used has not been found.")

    # converting string paths to Path instances
    training_src_file = Path(training_src_file)
    patterns_file_path = Path(patterns_file_path)

    logger.debug(f"Taking data from: {training_src_file}")
    logger.debug(f"Taking patterns from: {patterns_file_path}")

    assert isinstance(training_src_file, Path) and training_src_file is not None, f"{training_src_file} is not Path instance or is None"
    assert isinstance(patterns_file_path, Path) and patterns_file_path is not None, f"{patterns_file_path} is not Path instance or is None"

    train_data_with_overlaps = launch_auto_annotation(
        patterns_file_path,
        label_text,
        training_src_file,
        model,
        extract_texts,
        use_phrasematcher
    )

    final_train_data = []  # list will contain training data _without_ overlaps

    for text, annotations in train_data_with_overlaps:
        new_annotations = train.remove_overlaps(annotations)
        final_train_data.append((text, new_annotations))  # Saving modified sorted annotations without overlaps

    # training_src_file is checked above to be not None
    if not save:
        print(*final_train_data, sep="\n")
        return final_train_data

    if save_to is None:
        logger.info("Overwriting original training file")
        save_to_file = training_src_file
    else:
        logger.info(f"'{save_to}' path will be used to save the result.")
        save_to_file = Path(save_to)

    with save_to_file.open(mode="w") as f:
        json.dump(final_train_data, f)

    entity_trimmer(save_to_file)
    train.pretty_print_training_data(save_to_file)

    return final_train_data


def launch_auto_annotation(
    patterns_file_path: Path,
    label_text: str,
    training_src_file: Path,
    model="en_core_web_md",
    extract_texts: bool = False,
    use_phrasematcher: bool = False
):
    """Automatic annotation tool. The function takes a file which has to contain a
    JSON list of rules to be matched. The rules are in the format compatible
    with spaCy Matcher or PhraseMatcher objects. Rule recognition is done by
    spaCy pattern matching in the given text.

    :type patterns_file_path: Path
    :param patterns_file_path: String representing a path to the file with
        words to be matched (glossary etc).
    :type label_text: str
    :param label_text: The text of the label of an entity.
    :type model: str
    :param model: Model to be loaded to spaCy. Either a valid spaCy pre-trained
        model or a path to a local model.
    :type training_src_file: Path
    :param training_src_file: Training data source file path. JSON file is supposed to
        contain list of (text, annotations) tuples, where the text is the string
        and annotations represents a dictionary with list of (start, end, label)
        entity descriptors.
    :type extract_texts: bool
    :param extract_texts: A flag indicating whether new texts should be searched
        for.
    :type use_phrasematcher: bool
    :param use_phrasematcher: A flag indicating whether Matcher or PhraseMatcher
        spaCy object is used.
    """

    # almost result list - list containing all entities - including the overlaps
    train_data_with_overlaps = []
    training_src_file = training_src_file.resolve()
    patterns_file_path = patterns_file_path.resolve()

    texts, entities = get_current_texts_and_ents(training_src_file, extract_texts)

    nlp = spacy.load(model)  # create NLP analyzer object of the model
    with patterns_file_path.open(mode="r") as pttrns_file:
        patterns = json.load(pttrns_file)  # phrase/patterns to be matched

    if use_phrasematcher:
        # create PhraseMatcher object
        matcher = PhraseMatcher(nlp.vocab, attr="ORTH", validate=True)
        # process the keys and store their values in the patterns list
        keywords = list(nlp.pipe(patterns))
        # add all patterns to the matcher
        matcher.add(label_text, keywords)
    else:
        # create Matcher object
        matcher = Matcher(nlp.vocab, validate=True)
        matcher.add(label_text, patterns)

    logger.info(f"Using {matcher}")

    for doc in nlp.pipe(texts, batch_size=256):
        matches = matcher(doc)
        matched_spans = [doc[start:end] for match_id, start, end in matches]

        if matched_spans:
            logger.info(f"Doc index: {texts.index(doc.text)}, Matched spans {len(matched_spans)}: {matched_spans}")
        new_entities = [(span.start_char, span.end_char, label_text) for span in matched_spans]
        # following line of code also resolves situation when the entities dictionary is None
        train_example = (doc.text, {"entities": new_entities})
        if entities is not None:
            doc_index = texts.index(doc.text)
            old_entities = list(entities[doc_index]["entities"])
            new_entities += old_entities
            train_example = (doc.text, {"entities": new_entities})

        train_data_with_overlaps.append(train_example)

    return train_data_with_overlaps


def manual_annotation_handler(
        file_path: str,
        labels_path: str,
        lines: int = -1,
        start_index: int = 0,
        save: bool = True
):
    """

    :type labels_path:  str, list
    :param labels_path: Path to the file containing available entity labels or
                        the directly the list containing the labels names.
    :type file_path:    str
    :param file_path:   The path to the file containing texts to be annotated.
                        If None, then a user can write own sentences and
                        annotate them.
    :type lines:        int
    :param lines:       The number of texts to be annotated (1 text = 1 line),
                        defaults to -1 - means all the lines.
    :type start_index:    int
    :param start_index:   The index of the first text to be annotated.
    :type save:         bool
    :param save:        A flag indicating whether the result of the annotation
                        should be saved.
    :return:
    """
    file_path = Path(file_path)
    labels_path = Path(labels_path)

    assert file_path is not None, "file_path is None"
    assert labels_path is not None, "labels_path is None"

    labels = list(get_entities(labels_path).keys())

    try:
        if file_path.exists():
            if file_path.suffix == ".csv":
                extractor = CsvAsrsDataExtractor([file_path])
                texts = extractor.get_narratives(lines_count=lines, start_index=start_index)
            else:
                with file_path.open(mode="r") as file:
                    texts = json.load(file)
        else:
            # use given argument as the text to be annotated
            texts = [str(file_path)]
    except OSError:
        # use given argument as the text to be annotated
        texts = [str(file_path)]
        print()  # print an empty line

    # if we don't want to annotate all texts
    if lines != -1:
        texts = texts[start_index: start_index + lines]

    result = []
    for train_data in launch_man_annotation(texts, labels):
        result.append(train_data)
        if save:
            train_data_file = Path("data_files", "ner", "train_data", "annotated_" + file_path.name).resolve()
            logger.debug(train_data_file)
            train_data_file.touch(exist_ok=True)

            # if the file is not empty
            if len(train_data_file.read_bytes()) != 0:
                # rewrite the current content of the file
                with open(os.path.expanduser(train_data_file), mode="r") as file:
                    old_content = json.load(file)
            else:
                old_content = []

            with open(os.path.expanduser(train_data_file), mode="w") as file:
                old_content.append(train_data)
                json.dump(old_content, file)
                logger.info(f"Content in the {train_data_file.name} updated.\n")

            train.pretty_print_training_data(train_data_file)

    return result


def launch_man_annotation(texts: list, labels: list):
    """
    Manual text annotation tool. A set of texts from file_path parameter
    starting with start_index is progressively printed in order to be annotated
    by labels given in the labels_path.

    :param texts        Texts
    :param labels       Labels
    :return:            List of texts and its annotations.
    """
    # from avisaf.util.data_extractor import get_entities, get_narratives

    for text in texts:
        ent_labels = []
        print(text, "", sep="\n")
        words = input("Write all words that should be annotate (separated by a comma): ")
        spans = set([word.strip() for word in words.split(",") if word.strip()])

        if not spans:
            new_entry = (text, {"entities": []})
            yield new_entry
            # result.append(new_entry)
        else:
            # find positions of "spans" string list items in the text
            found_occurs = get_spans_indexes(text, list(spans))
            for occur_dict in found_occurs:
                key = list(occur_dict.keys())[0]  # only the first key is desired
                matches = occur_dict[key]
                if len(labels) == 1:
                    ent_labels += [(start, end, labels[0]) for start, end in matches]  # using automatic annotations if only one entity is defined
                    continue

                label = input(
                    f"Label '{key}' with an item from: {list(enumerate(labels))} or type 'NONE' to skip: "
                ).upper()
                if label not in labels and not label.isdigit():  # when there is no suitable label in the list
                    continue

                if label.isdigit():
                    ent_labels += [(start, end, labels[int(label)]) for start, end in matches]  # create the tuple
                else:
                    # same as above, but entity label text is directly taken
                    ent_labels += [(start, end, label) for start, end in matches]

            ents_no_overlaps = train.remove_overlaps({"entities": ent_labels})

            new_entry = (text, ents_no_overlaps)
            # result.append(new_entry)
            yield new_entry
        # print()  # print an empty line
