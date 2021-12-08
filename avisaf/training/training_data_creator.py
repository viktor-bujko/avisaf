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
import logging
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher

# importing own modules used in this module
from avisaf.util.indexing import get_spans_indexes, entity_trimmer
import avisaf.util.training_data_build as train
from avisaf.util.data_extractor import DataExtractor, get_narratives, get_entities
import avisaf.classification.vectorizers as vectorizers
from sklearn.preprocessing import LabelEncoder
import numpy as np


# looking for the project root
path = Path(__file__)
while not str(path.resolve()).endswith("avisaf"):
    path = path.parent.resolve()

SOURCES_ROOT_PATH = Path(path).resolve()
if str(SOURCES_ROOT_PATH) not in sys.path:
    sys.path.append(str(SOURCES_ROOT_PATH))


def get_current_texts_and_ents(train_data_file: Path, extract_texts: bool):
    if extract_texts:
        # get testing texts
        texts = list(get_narratives(train_data_file))
        entities = None
    else:
        with train_data_file.open(mode="r") as tr_data_file:
            # load the JSON file containing the list of training ('text string', entity dict) tuples
            texts, entities = zip(*json.load(tr_data_file))  # replacement for the code below

            # texts, entities = [], []
            # for text, ents in json.load(tr_data_file):
            #    texts.append(text)
            #    entities.append(ents)

    return texts, entities


def ner_auto_annotation_handler(
        patterns_file_path: str,
        label_text: str,
        training_src_file: str,
        model="en_core_web_md",
        extract_texts: bool = False,
        use_phrasematcher: bool = False,
        save: bool = False,
        save_to: str = None,
        verbose: bool = False
):
    """

    :param verbose:
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

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    if training_src_file is None:
        logging.error("The training data src file path cannot be None")

    if patterns_file_path is None:
        logging.error("File with patterns supposed to be used has not been found.")

    # converting string paths to Path instances
    training_src_file = Path(training_src_file)
    patterns_file_path = Path(patterns_file_path)

    logging.debug(f"Taking data from: {training_src_file}")
    logging.debug(f"Taking patterns from: {patterns_file_path}")

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
        final_train_data.append(
            (text, new_annotations)  # Saving modified sorted annotations without overlaps
        )

    # training_src_file is checked above to be not None
    if not save:
        print(*final_train_data, sep="\n")
        return final_train_data

    if save_to is None:
        logging.info("Overwriting original training file")
        save_to_file = training_src_file
    else:
        logging.info(f"'{save_to}' path will be used to save the result.")
        save_to_file = Path(save_to)

    with save_to_file.open(mode="w") as f:
        json.dump(final_train_data, f)

    # train.remove_overlaps_from_file(training_src_file)
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

    logging.info(f"Using {matcher}")

    for doc in nlp.pipe(texts, batch_size=256):
        matches = matcher(doc)
        matched_spans = [doc[start:end] for match_id, start, end in matches]

        if matched_spans:
            logging.info(f"Doc index: {texts.index(doc.text)}, Matched spans {len(matched_spans)}: {matched_spans}")
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


def ner_man_annotation_handler(
        file_path: str,
        labels_path: [str, list] = None,
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
    if not isinstance(labels_path, list):
        labels_path = Path(labels_path)

    assert file_path is not None, "file_path is None"
    assert labels_path is not None, "labels_path is None"

    if isinstance(labels_path, list):
        labels = labels_path
    else:
        labels = list(get_entities(labels_path).keys())

    try:
        if file_path.exists():
            if file_path.suffix == ".csv":
                texts = get_narratives(lines_count=lines, file_path=file_path, start_index=start_index)
            else:
                with file_path.open(mode="r") as file:
                    texts = json.load(file)
        else:
            texts = [str(file_path)]
    except OSError:
        # use given argument as the text to be annotated
        texts = [str(file_path)]
        print()  # print an empty line

    # if we don't want to annotate all texts
    if lines != -1:
        # TODO: texts accessed before assignment
        texts = texts[start_index: start_index + lines]

    result = []
    for train_data in launch_man_annotation(texts, labels):
        result.append(train_data)
        if save:
            train_data_file = Path("data_files", "ner", "train_data", "annotated_" + file_path.name).resolve()
            logging.debug(train_data_file)
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
                print(f"Content in the {train_data_file.relative_to(SOURCES_ROOT_PATH.parent)} updated.\n")

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


class ASRSReportDataPreprocessor:
    def __init__(self, vectorizer=None, encoders=None):
        self._label_encoders = [] if not encoders else encoders
        # self.vectorizer = vectorizers.TfIdfAsrsReportVectorizer() if vectorizer is None else vectorizer
        # self.vectorizer = vectorizers.SpaCyWord2VecAsrsReportVectorizer() if vectorizer is None else vectorizer
        # self.vectorizer = vectorizers.GoogleNewsWord2VecAsrsReportVectorizer()
        self.vectorizer = (
            vectorizers.Doc2VecAsrsReportVectorizer()
            if vectorizer is None
            else vectorizer
        )
        # self.vectorizer = vectorizers.FastTextAsrsReportVectorizer()

    def filter_texts_by_label(
        self,
        src_dict: dict,
        texts: np.ndarray,
        target_labels: list,
        target_label_filter: list = None,
        train: bool = False,
    ):
        """
        :param train:
        :param src_dict:
        :param target_label_filter:
        :param texts:
        :param target_labels:
        :return:
        """

        new_texts, new_labels = [], []
        """target_labels = np.array(list(src_dict.values()))

        for idx in range(len(src_dict.keys())):
            matching_texts_mask, matching_texts_labels = [], []
            for filter_pttrn in target_label_filter[idx]:
                pattern_mask = []

                def matches_filter(sample_target):
                    return sample_target.startswith(filter_pttrn) or sample_target.endswith(filter_pttrn)
                # iterating through all values to filter - this also preserves labels composition
                for i, sample_targets in enumerate(map(lambda lbl: lbl.split('; '), target_labels[idx])):         # Takes different values in the given column
                    if any(map(matches_filter, sample_targets)):
                        pattern_mask.append(i)
                matching_texts_mask.append(texts[np.array(pattern_mask)])
                matching_texts_labels.append(np.full(shape=len(pattern_mask), fill_value=filter_pttrn))
            new_texts.append(
                # concatenates all the texts and labels into (n_texts, 2) where n_texts corresponds to
                # all texts with corresponding filtered labels

                # will be used for simpler shuffling
                np.concatenate([
                    np.reshape(np.concatenate(matching_texts_mask), (-1, 1)),
                    np.reshape(np.concatenate(matching_texts_labels), (-1, 1))
                ], axis=1)
            )

        state = np.random.get_state()
        for text_data in new_texts:
            np.random.shuffle(text_data)
            np.random.set_state(state)
            np.random.shuffle(target_labels)"""

        for text_idx, text in enumerate(texts):
            # Some reports may be annotated with multiple labels separated by ;
            # We want to use them all as possible outcomes
            labels = target_labels[text_idx].split("; ")

            for label in labels:
                label = label.strip()
                matched_label = label  # label which will be used as prediction label

                if target_label_filter is not None:
                    # apply label filtration and truncation
                    matched_filter = list(
                        filter(
                            lambda x: str(label).startswith(x.strip()),
                            target_label_filter,
                        )
                    ) + list(
                        filter(
                            lambda x: str(label).endswith(x.strip()),
                            target_label_filter,
                        )
                    )

                    if len(matched_filter) != 1:

                        if label in matched_filter:
                            matched_label = label
                        else:
                            # The label is not unambiguous
                            continue
                    else:
                        matched_label = matched_filter[0]
                new_texts.append(text)
                new_labels.append(matched_label)

        if target_label_filter:
            for label in target_label_filter:
                if label not in new_labels:
                    print(
                        f'The label has not been found. Check whether "{label}" is correct category spelling.',
                        file=sys.stderr,
                    )

        """result = []
        for idx, text in enumerate(new_texts):
            texts, labels = text[:, 0], text[:, 1]
            if train:
                encoder = LabelEncoder()
                labels = encoder.fit_transform(labels)
                self._label_encoders.append(encoder)
            else:
                encoder = self._label_encoders[idx]
                labels = encoder.transform(labels)
            _, counts = np.unique(labels, return_counts=True)
            logging.info(dict(zip(encoder.classes_, counts)))
            result.append((texts, labels))"""

        encoder = LabelEncoder()
        new_labels = encoder.fit_transform(new_labels)
        self._label_encoders.append(encoder)

        # return result
        return np.array(new_texts), new_labels

    def undersample_data_distribution(
        self, text_data, target_labels, deviation_percentage: float
    ):
        """

        :param deviation_percentage:
        :param text_data:
        :param target_labels:
        :return:
        """

        more_present_idxs, distribution_counts = self.get_most_present_idxs(
            target_labels
        )
        examples_to_remove = int(
            np.sum(
                (distribution_counts[more_present_idxs] - np.min(distribution_counts))
                * deviation_percentage
            )
        )

        repeated_match = 0
        filtered_indices = set()
        not_filtered_indices = set()
        while examples_to_remove > 0:
            rnd_index = np.random.randint(0, text_data.shape[0])
            should_be_filtered = target_labels[rnd_index] in more_present_idxs
            if not should_be_filtered:
                not_filtered_indices.add(rnd_index)
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
            examples_to_remove = (
                examples_to_remove - 1
                if (repeated_match > 0 and repeated_match % 100 == 0)
                else examples_to_remove
            )

        arr_filter = [idx not in filtered_indices for idx in range(text_data.shape[0])]

        return text_data[arr_filter], target_labels[arr_filter]

    def oversample_data_distribution(
        self, text_data, target_labels, deviation_percentage: float
    ):
        distribution_counts, dist = self.get_data_distribution(target_labels)
        most_even_distribution = 1 / len(distribution_counts)
        more_present_labels = np.where(dist > most_even_distribution)

        least_present_labels = np.concatenate(
            np.argwhere(dist < most_even_distribution)
        )

        examples_to_have_per_minor_class = int(
            np.mean(distribution_counts[more_present_labels]) * 0.85
        )  # (total_examples_to_add - np.sum(least_present_labels)) / least_present_labels.shape[0]

        for label in least_present_labels:
            to_add_per_class = (
                examples_to_have_per_minor_class - distribution_counts[label]
            )  # subtracting the number of examples we already have
            texts_filtered_by_label = text_data[target_labels == label]
            labels_filtered = target_labels[target_labels == label]
            # randomly choose less present data and its label
            idxs = np.random.randint(0, labels_filtered.shape[0], size=to_add_per_class)

            text_data = np.concatenate([text_data, texts_filtered_by_label[idxs]])
            target_labels = np.concatenate([target_labels, labels_filtered[idxs]])

        state = np.random.get_state()
        np.random.shuffle(text_data)
        np.random.set_state(state)
        np.random.shuffle(target_labels)

        if logging.INFO:
            distribution_counts, _ = self.get_data_distribution(target_labels)
            info_dict = {}
            for label, counts in zip(set(target_labels), distribution_counts):
                info_dict.update({label: counts})
            logging.info(f"New data distribution: { info_dict }")

        return text_data, target_labels

    def get_most_present_idxs(self, target):

        distribution_counts, dist = self.get_data_distribution(target)
        most_even_distribution = 1 / len(distribution_counts)
        return np.where(dist > most_even_distribution)[0], distribution_counts

    def normalize(self, text_data, target_labels, deviation_rate: float):
        old_data_counts = text_data.shape[0]
        # text_data, target_labels = self.undersample_data_distribution(text_data, target_labels, deviation_rate)
        text_data, target_labels = self.oversample_data_distribution(
            text_data, target_labels, deviation_rate
        )

        new_data_counts = text_data.shape[0]
        logging.debug(self.get_data_distribution(target_labels)[1])

        if old_data_counts - new_data_counts > 0:
            logging.info(
                f"Normalization: {old_data_counts - new_data_counts} of examples had to be removed to have an even distribution of examples"
            )

        return text_data, target_labels

    def vectorize_texts(
        self,
        texts_paths: list,
        labels_to_extract: list,
        train: bool,
        label_values_filter: list,
        normalize: bool = False,
    ):
        narrative_label = "Report 1_Narrative"

        extractor = DataExtractor(texts_paths)
        labels_to_extract = (
            labels_to_extract if labels_to_extract is not None else [narrative_label]
        )
        extracted_dict = extractor.extract_from_csv_columns(labels_to_extract)
        narratives = extractor.extract_from_csv_columns([narrative_label])[
            narrative_label
        ]

        logging.debug(labels_to_extract)
        logging.debug(label_values_filter)

        texts_labels_arr = []
        for idx, key in enumerate(extracted_dict.keys()):
            if label_values_filter:
                texts_labels_arr_1 = self.filter_texts_by_label(
                    extracted_dict,  # extracted_dict[narrative_label],
                    narratives,  # extracted_dict[labels_to_extract],
                    extracted_dict[key],  # labels_to_extract,
                    label_values_filter[idx],  # label_values_filter,
                    train=train,
                )
                texts_labels_arr.append(texts_labels_arr_1)
            else:
                texts_labels_arr_1 = self.filter_texts_by_label(
                    extracted_dict, narratives, extracted_dict[key], None, train=train
                )
                texts_labels_arr.append(texts_labels_arr_1)

        result_data, result_targets = [], []
        for texts, target_labels in texts_labels_arr:
            data = self.vectorizer.build_feature_vectors(
                texts, target_labels, train=train  # .shape[0],
            )

            vectorizers.show_vector_space_3d(data, target_labels)

            if normalize:
                data, target_labels = self.normalize(data, target_labels, 0.1)

            result_data.append(data)
            result_targets.append(target_labels)

        return np.array(result_data), np.array(result_targets)

    @staticmethod
    def get_data_distribution(target: list):
        """

        :param target:
        :return:
        """

        distribution_counts = np.histogram(target, bins=np.unique(target).shape[0])[0]
        dist = distribution_counts / np.sum(
            distribution_counts
        )  # Gets percentage presence of each class in the data

        return distribution_counts, dist

    def encoder(self, idx):
        return self._label_encoders[idx]

    @property
    def encoders(self):
        return self._label_encoders
