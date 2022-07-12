import re

import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import LabelEncoder

from avisaf.ner.annotator import logger
from .vectorizers import VectorizerFactory


class ASRSReportDataPreprocessor:
    def __init__(self, vectorizer=None, encoders=None):
        self.label_encoders = {} if not encoders else encoders
        self._normalization_methods = {
            "undersample": self._undersample_data_distribution,
            "oversample": self._oversample_data_distribution
        }
        self.vectorizer = VectorizerFactory.create_vectorizer(vectorizer)

    def extract_labeled_data(self, extractor, labels_to_extract: list, set_default: dict, label_classes_filter: list = None, normalize: str = None):

        data = self.vectorizer.vectorize_texts(extractor)

        labels_to_extract = labels_to_extract if labels_to_extract is not None else []
        extracted_dict = extractor.extract_data(labels_to_extract)

        logger.debug(labels_to_extract)
        logger.debug(label_classes_filter)

        filtered_arrays = self._filter_texts_by_label(extracted_dict, label_classes_filter, set_default)
        extracted_data, targets = [], []
        for filtered_array in filtered_arrays:
            logger.debug(f"Filtered array shape: {filtered_array.shape}")
            text_idx_filter = (filtered_array[:, 0]).astype(np.int)  # ndarray with (text_index, text_label, text_encoded_label) items
            labels = np.reshape((filtered_array[:, -1]).astype(np.int), (-1, 1))
            # keeping labels separated from the text vectors
            extracted_data.append(data[text_idx_filter])
            targets.append(labels)

        extracted_data, targets = np.array(extracted_data, dtype=object), np.array(targets, dtype=object)
        # vectorizers.show_vector_space_3d(texts_labels_pairs)

        norm_method = self._normalization_methods.get(normalize)
        if not norm_method:
            if normalize:
                logger.warning(f"{normalize} normalization method is not supported. Please choose from: {list(self._normalization_methods.keys())}")
            # return extracted data without further modifications
            return extracted_data, targets

        return self._normalize(extracted_data, targets, norm_method)

    def get_data_targets_distribution(self, data_targets: list, label: str):
        """

        :param label:
        :param data_targets:
        :return:
        """
        label_encoder = self.label_encoders.get(label)
        if not label_encoder:
            logger.error(f"LabelEncoder object could not be found for \"{label}\".")
            raise ValueError()

        distribution_counts, _ = np.histogram(data_targets, bins=len(label_encoder.classes_))
        dist = distribution_counts / np.sum(distribution_counts)  # Gets percentage presence of each class in the data
        # named distribution - target classes names as keys
        dist = dict(zip(label_encoder.classes_, dist))

        return distribution_counts, dist

    def _filter_texts_by_label(self, extracted_labels: dict, target_label_filters: list, set_default_class: dict):
        """
        :param target_label_filters: List of lists for each extracted topic. Each list inside target_label_filters
                                     list contains desired class names which should be filtered for classification.
                                     Texts which are annotated with a label different from those in a given topic
                                     filter list, will be omitted from classification process.
        :param extracted_labels: Represents extracted data dictionary with extracted topic names as keys and
                                 array with target-class names annotations for each extracted text as values.
                                 The number of items contained in the dictionary equals the number of extracted
                                 topics, where each contains number_of_extracted_samples labels.
        :param set_default_class: Boolean flag which specifies whether texts, which do not correspond to any of the
                                  value defined in label_values list should still be included in training dataset
                                  with target label "Other".
        :return: Array of ndarrays for each topic to be classified. Each ndarray contains in first column the indices
                 of texts with corresponding target class label. Since each text may contain several matching target
                 classes for given classification topic, a text index may appear multiple times - once for each match.
                 Second and third column represent matching target class labels - string version in 2nd column and its
                 integer encoding in the 3rd column. Also, label encoder-decoder objects are set in this method.
        """
        target_labels = np.array(list(extracted_labels.values()))

        result_arrays = [[] for _ in range(target_labels.shape[0])]  # creating an empty array for each investigated topic
        encoders = dict(zip(extracted_labels.keys(), [LabelEncoder()] * len(extracted_labels.keys())))

        for text_idx in range(target_labels.shape[1]):
            # iterating through texts
            for idx, topic_label in enumerate(extracted_labels.keys()):  # iterating through different topics target classes
                label_set = extracted_labels.get(topic_label)[text_idx]
                class_filter_regexes = []
                if target_label_filters:
                    # override default empty list
                    class_filter_regexes = [re.compile(class_regex) for class_regex in target_label_filters[idx]]
                # Some reports may be correspond to multiple target classes separated by ";"
                # We want to use them all as possible outcomes
                labels = label_set.split(";")
                for label in labels:
                    label = label.strip()
                    if not class_filter_regexes:
                        result_arrays[idx].append(np.array([text_idx, label]))  # adding each label - not applying any filter
                        continue

                    matched_classes = []  # matched_classes contains all regexes that fit target classes
                    for class_regex in class_filter_regexes:
                        matched_regex = re.findall(class_regex, label)
                        if not matched_regex:
                            continue
                        matched_classes.append(class_regex.pattern)  # expecting only 1-item matched_regex list
                    if not matched_classes:
                        # target class did not match any desired filter item -> text will not be used for classification
                        if set_default_class.get(topic_label, False):
                            result_arrays[idx].append(np.array([text_idx, "Other"]))
                        continue
                    matched_class = max(matched_classes, key=len)  # taking longest matching pattern as target class
                    result_arrays[idx].append(np.array([text_idx, matched_class]))
                    break

        for idx, (result_array, encoder) in enumerate(zip(result_arrays, encoders.values())):
            target_classes = np.array(result_array)[:, 1]
            encoded_classes = np.reshape(encoder.fit_transform(target_classes), (-1, 1))
            result_arrays[idx] = np.concatenate([result_array, encoded_classes], axis=1)

        result_arrays = np.array([np.array(res_array) for res_array in result_arrays], dtype=object)
        self.label_encoders.update(encoders)

        return result_arrays

    def _get_most_present(self, topic_label: str, target_labels):

        distribution_counts, dist = self.get_data_targets_distribution(target_labels, label=topic_label)
        dist = np.array(list(dist.values()))  # not using label classes names now
        most_even_distribution = 1 / len(distribution_counts)
        return np.where(dist > most_even_distribution), distribution_counts, dist

    def _undersample_data_distribution(self, topic_label: str, text_data: np.ndarray, target_labels: np.ndarray):
        """

        :param topic_label:
        :param text_data:
        :param target_labels:
        :return:
        """
        more_present_idxs, distribution_counts, _ = self._get_most_present(topic_label, target_labels)
        normalized_counts = (np.min(distribution_counts) * np.random.uniform(low=1.05, high=1.2, size=len(distribution_counts))).astype(np.int)  # generating random overweight factor for each class
        examples_to_remove_counts = np.maximum(distribution_counts - normalized_counts, 0)  # preventing samples repetition by replacing negative number of removed samples by 0

        filtered_indices = set()
        for idx, to_remove_class_count in enumerate(examples_to_remove_counts):
            repeated_match = 0
            while to_remove_class_count > 0:
                text_idx = np.random.randint(0, text_data.shape[0])  # take random text sample
                if target_labels[text_idx] != idx:
                    # filtration does not apply for different labels
                    continue

                # text_idx should be filtered
                if text_idx in filtered_indices:
                    repeated_match += 1
                else:
                    filtered_indices.add(text_idx)
                    to_remove_class_count -= 1
                    repeated_match = 0
                    continue

                # Avoiding "infinite loop" caused by always matching already filtered examples
                # Giving up on filtering the exact number of examples -> The distribution will be less even
                assert repeated_match > 0
                if repeated_match % 500 == 0:
                    to_remove_class_count -= 1
                    repeated_match = 0

        arr_filter = [idx not in filtered_indices for idx in range(text_data.shape[0])]

        return text_data[arr_filter], target_labels[arr_filter]

    def _oversample_data_distribution(self, topic_label: str, text_data: np.ndarray, target_labels: np.ndarray):
        """

        :param topic_label:
        :param text_data:
        :param target_labels:
        :return:
        """
        more_present_labels, distribution_counts, dist = self._get_most_present(topic_label, target_labels)
        normalized_counts = (np.mean(distribution_counts[more_present_labels]) * np.random.uniform(low=0.8, high=0.95, size=len(distribution_counts))).astype(np.int)
        examples_to_add_counts = np.maximum(normalized_counts - distribution_counts, 0)

        for label, to_add_class_count in enumerate(examples_to_add_counts):  # enumerate idx acts as label now
            texts_filtered_by_label = text_data[np.array(target_labels == label).ravel()]
            labels_filtered = target_labels[target_labels == label]
            # randomly choose less present data and its label
            idxs = np.random.randint(0, labels_filtered.shape[0], size=to_add_class_count)

            if type(text_data) == np.ndarray:
                text_data = np.concatenate([text_data, texts_filtered_by_label[idxs]])
            elif type(text_data) == sp.csr_matrix:
                text_data = sp.vstack([text_data, texts_filtered_by_label[idxs]])
            target_labels = np.concatenate([target_labels.ravel(), labels_filtered[idxs]])

        rng = np.arange(text_data.shape[0])  # index-based reshuffling
        np.random.shuffle(rng)
        text_data = text_data[rng, :]
        target_labels = target_labels[rng]

        return text_data, target_labels

    def _normalize(self, data: np.ndarray, targets: np.ndarray, normalization_method):

        normalized_extracted_data, normalized_targets = [], []
        for data, target_labels, topic_label_name in zip(data, targets, self.label_encoders.keys()):
            normalized_data, normalized_target_labels = normalization_method(topic_label_name, data, target_labels)
            logger.debug(f"Before normalization: {self.get_data_targets_distribution(target_labels, label=topic_label_name)[1]}")
            logger.debug(f"After normalization: {self.get_data_targets_distribution(normalized_target_labels, label=topic_label_name)[1]}")
            samples_diff = np.abs(data.shape[0] - normalized_data.shape[0])
            logger.info(f"Normalization: {samples_diff} examples had to be removed or added to obtain more even distribution of samples.")
            normalized_extracted_data.append(normalized_data)
            normalized_targets.append(normalized_target_labels)
        normalized_extracted_data, normalized_targets = np.array(normalized_extracted_data, dtype=object), np.array(normalized_targets, dtype=object)

        return normalized_extracted_data, normalized_targets

    def encoder(self, encoder_name: str):
        return self.label_encoders.get(encoder_name)

    @property
    def normalization_methods(self):
        return list(self._normalization_methods.keys())
