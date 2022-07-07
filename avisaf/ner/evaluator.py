#!/usr/bin/env python3

import json
import numpy as np
import logging
from pathlib import Path
import spacy
from util.data_extractor import find_file_by_path, get_entities

logger = logging.getLogger("avisaf_logger")


def evaluate_ner(model: str, texts_file: str):

    if not model:
        logger.error("Model has to be defined")
        return

    if not texts_file:
        logger.error("Texts file has to be defined")
        return

    model = Path(model)
    texts_file = Path(texts_file)

    StrictEvaluator(model).evaluate_texts_from_file(texts_file)
    # IntersectionEvaluator(model).evaluate_texts_from_file(texts_file)


class Evaluator:

    def __init__(self, model_to_evaluate: Path):
        self._nlp = spacy.load(model_to_evaluate)
        logger.debug(f"Current pipeline components: {self._nlp.pipe_names}")
        self._available_entities = get_entities().keys()
        self._annotated_count = 0  # number of annotated texts
        self._used_metrics = {
            "Precision": 0.0,
            "Recall": 0.0,
            "Accuracy": 0.0,
            "F1 Score": 0.0
        }

    def _initialize_stats_dicts(self) -> dict:
        """
        Creates a structure for each entity label as well as for the entire text.
        Each structure holds scores for (in)correct predictions for given entity.

        :return: The dictionary with available entity labels as keys and created
                stats-holding structures as their values.
        """

        entity_stats = {}
        stats = {
            "tp": 0.0,  # entity matches exactly, label is correct
            "fp": 0.0,  # entity has been found, but is not in the gold list -> false positive / spurious
            "tn": 0.0,  # entity has correctly not been found = total - (tp + fp + fn)
            "fn": 0.0,  # entity should have been found, but is not -> false negative
            "predicted": 0,
            "gold": 0,
            "tp_partial": 0.0,  # entity range match is partial, label is correct
            "incorrect": 0.0,  # entity ranges match, label is incorrect
            "total": 0  # all considered document tokens
        }

        entity_stats["TEXT"] = stats.copy()
        for entity in self._available_entities:
            entity_stats[entity] = stats.copy()

        return entity_stats

    @staticmethod
    def entities_list_to_dict(entity_tuples: list) -> dict:
        """
        Method which converts the list of (start_char, end_char, label) entities into a
        dictionary.

        :param entity_tuples: The list containing (start_char, end_char, label) tuples
                              which represents an entity.
        :return: Entity tuples list converted to the dictionary with (start_char, end_char)
                              tuple as key for each entity and label as its value.
        """
        entities = {}
        for ent_start, ent_end, ent_label in entity_tuples:
            # for given span of characters (ent_start, ent_end) at most 1 entity may be defined -> using the span as key
            entities[(ent_start, ent_end)] = ent_label
        return entities

    @staticmethod
    def spans_to_ent_tuples(entity_spans) -> list:
        """
        Method which converts spaCy's entity span objects into (start_char, end_char, label)
        tuples which are then compared with the given target / gold tuples.

        :param entity_spans: spaCy Span object which represents an entity (see: https://spacy.io/api/span)
                             for more information.

        :return: List of recognized entities in (start_char, end_char, label) tuple format.
        """
        entity_tuples = []
        for ent_span in entity_spans:
            # doc.ents contains list of all entities, each entity has .start_char a .end_char attribute
            entity_tuples.append(
                (ent_span.start_char, ent_span.end_char, ent_span.label_)
            )

        return entity_tuples

    def _compute_scores(self, stats: dict, include_accuracy: bool = True, ent_label: str = None) -> np.array:
        """
        Compute the scores based on the given the prediction statistics for each
        entity label or entire text.

        :param stats: Dictionary structure containing the statistics for score computation.
        :param include_accuracy: Boolean flag to indicate whether accuracy should be included.
                                 Note that accuracy computation needs the number of "true negatives",
                                 which is not supported for individual entities.
                                 Therefore, individual entities don't have their accuracy computed.

        :return: A numpy array containing the scores.
        """
        pass

    def _collect_stats(self, predicted_list: list, gold_list: list) -> tuple:
        """
        Method which is used for collection of statistics about individual predicted entities
        as well as the entire text.

        :param predicted_list: List of predicted entities in (start_char, end_char, label) format.
        :param gold_list: List of target / gold entities in (start_char, end_char, label) format.

        :return: 2-tuple of dictionaries; 1st dictionary contains available entity labels as keys and
                statistics for given entity label as value. 2nd dictionary contains the statistics for
                entire text (see _initialize_stats_dicts() method for more details about stats-holding
                structure).
        """
        pass

    def _evaluate(self, document, gold_list: list) -> tuple:
        """
        Evaluate named entity recognition model on given text.

        :param document: spaCy Document object - model-processed representation of given text
        :param gold_list: The list of target / gold named entities.
        :return: Scores for the entire text as well as per-named entity scores.
        """
        predicted_list = self.spans_to_ent_tuples(document.ents)
        entity_stats, text_stats = self._collect_stats(predicted_list, gold_list)

        for token in document:
            if not token.is_punct:
                text_stats["total"] += 1

        text_stats["tn"] = np.max([
            text_stats["total"] - (text_stats["tp"] + text_stats["fp"] + text_stats["fn"]),
            0
        ])

        for stat_key, stat_value in text_stats.items():
            assert stat_value >= 0, f"The value for \"{stat_key}\" is {stat_value}"

        return text_stats, entity_stats

    def _aggregate_and_print_results(self, text_metrics: np.array, entity_class_metrics: np.array):
        for idx, metric in enumerate(self._used_metrics):
            logger.debug(f"Setting \"{metric}\"")
            self._used_metrics[metric] = text_metrics[idx]

        print(f"Average performance per {self._annotated_count} texts:")
        for metric in self._used_metrics:
            print("\t{}: {:.4f}".format(metric, self._used_metrics[metric]))
        for idx, ent_type in enumerate(self._available_entities):
            print(
                f"\t{ent_type}",
                "\t\tPrecision: {:.4f}".format(entity_class_metrics[idx, 0] * 100),
                "\t\tRecall: {:.4f}".format(entity_class_metrics[idx, 1] * 100),
                "\t\tF1 Score: {:.4f}".format(entity_class_metrics[idx, 2] * 100),
                sep="\n"
            )

    def evaluate_texts_from_file(self, texts_file: Path):
        """
        Wrapper around evaluation starter method which accepts a JSON file containing an
        array in (text, annotations) format, where 'text' represents the text
        have its named entities recognized and 'annotations' represent the array of
        target named entity annotations in (start_char, end_char, label) format.

        :param texts_file: Path to the file containing texts in the format described above.
        """
        file = find_file_by_path(texts_file)

        if not file:
            logger.error(f"File {texts_file} has not been found.")
            return

        with file.open(mode="r") as f:
            texts_to_evaluate = json.load(f)

        if not texts_to_evaluate or not isinstance(texts_to_evaluate, list):
            logger.warning("No suitable texts to be evaluated have been found.")
            return

        self.evaluate_texts(texts_to_evaluate)

    def evaluate_texts(self, texts_to_evaluate: list):
        """
        Evaluates named entity recognition of given list of (text, annotations) texts.

        :param texts_to_evaluate: List of (text, gold annotations) tuples.
        """
        total_entity_stats = self._initialize_stats_dicts()
        total_text_stats = total_entity_stats.pop("TEXT")

        txt_idx = 0
        all_texts, all_gold_entities = zip(*texts_to_evaluate)
        other_pipes = [pipe for pipe in self._nlp.pipe_names if pipe != "ner"]  # only named-entity recognizer will be used
        # processing the given batch of texts to get the predictions
        for txt_idx, doc in enumerate(self._nlp.pipe(all_texts, batch_size=512, disable=other_pipes)):
            if txt_idx % 100 == 0:
                # reducing the number of output messages
                print(f"Evaluating text {txt_idx + 1} / {len(texts_to_evaluate)}")
            text_stats, per_entity_stats = self._evaluate(doc, all_gold_entities[txt_idx]["entities"])
            for stat, value in text_stats.items():
                total_text_stats[stat] += value
            for entity_label, entity_stats in per_entity_stats.items():
                for stat, value in entity_stats.items():
                    total_entity_stats[entity_label][stat] += value

        metrics = self._compute_scores(total_text_stats, ent_label="Text")
        entity_class_metrics = []
        for entity_label, entity_stats in total_entity_stats.items():
            entity_class_metrics.append(self._compute_scores(entity_stats, include_accuracy=False, ent_label=entity_label))

        self._annotated_count = txt_idx + 1
        self._aggregate_and_print_results(np.array(metrics), np.array(entity_class_metrics))


class StrictEvaluator(Evaluator):

    def __init__(self, model_to_evaluate: Path):
        super().__init__(model_to_evaluate)

    @staticmethod
    def _get_confusion_matrix_items(gold: dict, predicted: dict):
        """
        Given gold / target and actually predicted named entity annotations,
        compute the number of true and false positives as well as false negatives.
        This method uses exact label and position match to determine whether an
        entity has been recognized correctly.

        :param gold: Dictionary which contains (start_char, end_char) target tuples as
                     named entity occurrence keys and label as value.
        :param predicted: Dictionary which contains (start_char, end_char) tuples as keys
                          of actual predicted named entities and labels as values.
        :return: Tuples in ((start_char, end_char), label) format for false negatives,
                 false positives and true positives.
        """
        gold_set = set(gold.items())
        predicted_set = set(predicted.items())
        # entities which should have been predicted, but are not -> false negatives
        false_negatives = gold_set.difference(predicted_set)
        # entities which are predicted by the model, but miss in the gold set -> false positives
        false_positives = predicted_set.difference(gold_set)
        # entities which are in both sets
        true_positives = predicted_set.intersection(gold_set)

        return false_negatives, false_positives, true_positives

    def _collect_stats(self, predicted_list: list, gold_list: list):
        """See base class docstring."""
        entity_stats = self._initialize_stats_dicts()
        text_stats = entity_stats.pop("TEXT")  # extracting overall text stats dict

        # "hashing" entities into "(start_char, end_char): label" dictionaries below
        predicted_dict = self.entities_list_to_dict(predicted_list)
        gold_dict = self.entities_list_to_dict(gold_list)

        false_negatives, false_positives, true_positives = self._get_confusion_matrix_items(gold_dict, predicted_dict)
        confusion_matrix = {
            "fn": false_negatives,
            "fp": false_positives,
            "tp": true_positives,
            "predicted": predicted_dict.items(),
            "gold": gold_dict.items()
        }

        self._set_stats_per_entity(entity_stats, confusion_matrix)
        for item_name, item in confusion_matrix.items():
            text_stats[item_name] = len(item)  # the number of tp items, fp items etc ..

        # completion of overall statistics by computing the number of true negatives
        text_stats["tn"] = np.max([
            text_stats["total"] - (text_stats["tp"] + text_stats["fp"] + text_stats["fn"]),
            0
        ])

        return entity_stats, text_stats

    @staticmethod
    def _set_stats_per_entity(entity_stats: dict, confusion_matrix: dict):
        # confusion matrix entry example:
        # key: "fp" - item_name
        # value: set of (start_char, end_char, label) tuples which were predicted as false positives
        for item_name, items in confusion_matrix.items():
            # items is a set of false/true_positives / false_negatives
            for ent_range, ent_label in items:
                entity_stats[ent_label][item_name] += 1

        return entity_stats

    def _compute_scores(self, stats: dict, include_accuracy: bool = True, ent_label: str = None) -> np.array:
        """See base class docstring."""

        if stats["predicted"] != 0:
            # Model has found at least one entity
            precision = stats["tp"] / stats["predicted"]
        else:
            # Model has found 0 entities -> this case is correct only if the model should not have found any annotations
            logger.warning(f"Undefined Precision Metric{f' for {ent_label}: ' if ent_label else ': '}")
            # precision = 1.0 if stats["gold"] == 0 else 0.0
            precision = 0.0

        if stats["gold"] != 0:
            recall = stats["tp"] / stats["gold"]
        else:
            # gold entities list contains 0 entities -> 0 should be found
            logger.warning(f"Undefined Recall Metric{f' for {ent_label}: ' if ent_label else ': '}")
            # recall = 1.0 if stats["predicted"] == 0 else 0.0
            recall = 0.0

        f1_score = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0.0

        if include_accuracy:
            if stats["total"] != 0:
                acc = (stats["tp"] + stats["tn"]) / (stats["total"])
            else:
                # Token contains 0 non-punctuation tokens.
                acc = np.nan
        else:
            acc = np.nan

        if include_accuracy:
            logger.debug("{:.4f} | {:.4f} | {:.4f} | {:.4f}".format(precision, recall, acc, f1_score))
        else:
            logger.debug("{:.4f} | {:.4f} | {} | {:.4f}".format(precision, recall, " " * 5, f1_score))

        result_scores = [precision, recall, acc, f1_score] if include_accuracy else [precision, recall, f1_score]
        for score in result_scores:
            assert 0.0 <= score <= 1.0, f"The \"{score}\" is unbound."

        return np.array(result_scores)


class IntersectionEvaluator(Evaluator):

    def __init__(self, model_to_evaluate: Path):
        super().__init__(model_to_evaluate)

    @staticmethod
    def _intersection_over_union(prediction_range: set, gold_range: set) -> float:
        """
        Computes intersection over union metric. Intersection is represented by the
        range of characters present in both predicted and gold named entity
        representation. Union is represented by the range of characters present in
        at least one (predicted or gold or both) entity representations.

        :param prediction_range: Range of characters which represent predicted named entity.
        :param gold_range: Range of characters which represent target named entity.
        :return: Intersection over union ratio.
        """
        ent_intersection = prediction_range.intersection(gold_range)
        ent_union = prediction_range.union(gold_range)

        iou = len(ent_intersection) / len(ent_union)
        assert 0 <= iou <= 1, f"IOU equals {iou}"
        return iou

    def _set_entity_match_score(self, prediction: tuple, gold_list: list, text_ent_stats: dict):
        """
        Compute the match score for a predicted named entity by comparing it to the
        target / gold list of named entities and update text_ent_stats accordingly.

        :param prediction: A single representation of named entity predicted by a model.
        :param gold_list:  The list of target / gold named entities.
        :param text_ent_stats: See Evaluator._initialize_stats_dicts() method for
                               detailed description of statistics holding structure.
        :return:
        """
        pred_start, pred_end, pred_label = prediction
        prediction_range = set(range(pred_start, pred_end))
        matching_gold = None
        entity_iou = 0.0
        for gold_start, gold_end, gold_label in gold_list:
            # searching the list of gold entities for a match
            gold_range = set(range(gold_start, gold_end))
            entity_iou = self._intersection_over_union(prediction_range, gold_range)
            if entity_iou > 0.0:
                # found overlapping entity
                matching_gold = (gold_start, gold_end, gold_label)
                break  # only first corresponding overlap is considered, because only one entity can include given span
        if matching_gold is None:
            # no matching gold entity has been found -> false positive prediction
            text_ent_stats[pred_label]["fp"] += 1
            text_ent_stats[pred_label]["predicted"] += 1
            return
            # return 0.0  # entity_iou = 0.0

        match_start, match_end, match_label = matching_gold
        assert entity_iou > 0.0, f"Non overlapping"
        text_ent_stats[pred_label]["predicted"] += 1  # prediction has been made
        text_ent_stats[pred_label]["gold"] += 1  # gold entity overlap exists
        if pred_label == match_label:
            text_ent_stats[pred_label]["tp"] += entity_iou
        else:
            text_ent_stats[pred_label]["incorrect"] += entity_iou

        # return entity_iou

    def _collect_stats(self, predicted_list: list, gold_list: list):
        """See base class docstring."""
        predicted_dict = self.entities_list_to_dict(predicted_list)
        gold_dict = self.entities_list_to_dict(gold_list)

        entity_stats = self._initialize_stats_dicts()
        text_stats = entity_stats.pop("TEXT")

        for prediction in predicted_list:
            # update text entity stats by each prediction
            self._set_entity_match_score(prediction, gold_list, entity_stats)

        false_negatives = set(gold_dict.items()).difference(set(predicted_dict.items()))
        # (start, end, label) tuples which are in gold list but not predicted
        for _, fn_label in false_negatives:
            entity_stats[fn_label]["fn"] += 1
            entity_stats[fn_label]["gold"] += 1

        text_stats = self._get_overall_text_stats(entity_stats, text_stats)
        return entity_stats, text_stats

    def _compute_scores(self, stats: dict, include_accuracy: bool = True, ent_label: str = None) -> np.array:
        """See base class docstring."""

        if stats["predicted"] != 0:
            # Model has found at least one entity
            relevant_entity_ranges = stats["tp"] + stats["tp_partial"] - stats["incorrect"]
            precision = 0.0 if relevant_entity_ranges < 0 else (relevant_entity_ranges / stats["predicted"])
        else:
            # Model has found 0 entities -> this case is correct only if the model should not have found any annotations
            precision = 1.0 if stats["gold"] == 0 else 0.0

        if stats["gold"] != 0:
            relevant_entity_ranges = stats["tp"] + stats["tp_partial"] + stats["incorrect"]
            # incorrect entity is still relevant - it has been found even though it has incorrect label
            recall = relevant_entity_ranges / stats["gold"]
        else:
            # gold entities list contains 0 entities -> 0 should be found
            recall = 1.0 if stats["predicted"] == 0 else 0.0

        f1_score = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0.0

        if include_accuracy:
            if stats["total"] != 0:
                acc = (stats["tp"] + stats["tn"]) / (stats["total"])
            else:
                # Token contains 0 non-punctuation tokens.
                acc = np.nan
        else:
            acc = np.nan

        if include_accuracy:
            logger.debug("{:.3f} | {:.3f} | {:.3f} | {:.3f}".format(precision, recall, acc, f1_score))
        else:
            logger.debug("{:.3f} | {:.3f} | {} | {:.3f}".format(precision, recall, " " * 5, f1_score))

        result_scores = [precision, recall, acc, f1_score] if include_accuracy else [precision, recall, f1_score]
        for score in result_scores:
            assert 0.0 <= score <= 1.0, f"The \"{score}\" is unbound."

        return np.array(result_scores)

    @staticmethod
    def _get_overall_text_stats(entity_stats: dict, txt_stats: dict):
        """
        Aggregate per-named entity statistics.
        :return: Modified values of text_stats structure.
        """
        for ent_label, ent_stats in entity_stats.items():
            for ent_stat, stat_value in ent_stats.items():
                txt_stats[ent_stat] += stat_value

        return txt_stats
