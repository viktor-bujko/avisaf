#!/usr/bin/env python3

import json
import logging
import numpy as np
from pathlib import Path
import spacy
from avisaf.util.data_extractor import find_file_by_path, get_entities


def evaluate_spacy_ner(model: str, texts_file: str):

    if not model:
        logging.error("Model has to be defined")
        return

    if not texts_file:
        logging.error("Texts file has to be defined")
        return

    model = Path(model)
    texts_file = Path(texts_file)

    StrictEvaluator(model).evaluate_texts_file(texts_file)


class Evaluator:

    def __init__(self, model_to_evaluate: Path):
        self._nlp = spacy.load(model_to_evaluate)
        self._available_entities = get_entities().keys()
        self._annotated_count = 0  # number of annotated texts

    @staticmethod
    def build_entity_dict(entity_tuples: list):
        """
        Method which converts the list of (start_char, end_char, label) entities into the
        dictionary for faster search.

        :param entity_tuples: The list containing the tuples (start_char, end_char, label)
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
        entity_tuples = []
        for ent_span in entity_spans:
            # doc.ents contains list of all entities, each entity has .start_char a .end_char attribute
            entity_tuples.append(
                (ent_span.start_char, ent_span.end_char, ent_span.label_)
            )

        return entity_tuples

    def _get_scores(self, stats):
        pass

    def _evaluate(self, document, gold_list: list) -> tuple:
        pass

    def _aggregate_and_print_results(self, text_metrics: np.array, entity_class_metrics: np.array):
        pass

    def evaluate_texts_file(self, texts_file: Path):
        file = find_file_by_path(texts_file)

        if not file:
            logging.error(f"File {texts_file} has not been found.")
            return

        with file.open(mode="r") as f:
            texts_to_evaluate = json.load(f)

        if not texts_to_evaluate or not isinstance(texts_to_evaluate, list):
            logging.warning("No suitable texts to be evaluated have been found.")
            return

        self.evaluate_texts(texts_to_evaluate)

    def evaluate_texts(self, texts_to_evaluate: list):
        metrics = []
        entity_class_metrics = []

        txt_idx = 0
        all_texts, all_gold_entities = zip(*texts_to_evaluate)
        # processing the given batch of texts to get the predictions
        for txt_idx, doc in enumerate(self._nlp.pipe(all_texts, batch_size=512)):
            if txt_idx % 100 == 0:
                # reducing the number of output messages
                print(f"Evaluating text {txt_idx + 1} / {len(texts_to_evaluate)}")
            text_metrics, per_entity_metrics = self._evaluate(doc, all_gold_entities[txt_idx]["entities"])
            metrics.append(text_metrics)
            entity_class_metrics.append(per_entity_metrics)

        self._annotated_count = txt_idx + 1
        self._aggregate_and_print_results(np.array(metrics), np.array(entity_class_metrics))


class StrictEvaluator(Evaluator):

    def __init__(self, model_to_evaluate: Path):
        super().__init__(model_to_evaluate)
        self._used_metrics = {
            "Precision": 0.0,
            "Recall": 0.0,
            "Accuracy": 0.0,
            "F1 Score": 0.0
        }

    def _initialize_stats_dicts(self):
        entity_stats = {}
        text_stats = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "predicted": 0,   # number of predicted entities
            "gold": 0,  # number of golden entities
            "total": 0
        }

        for entity in self._available_entities:
            entity_stats[entity] = {
                "tp": 0,  # entity matches exactly, label is correct
                # "tp_partial": 0,  # entity range match is partial, label is correct
                # "incorrect": 0,  # entity ranges match, label is incorrect
                "fp": 0,  # entity has been found, but is not in the gold list -> false positive / spurious
                "fn": 0,  # entity should have been found, but is not -> false negative
                "predicted": 0,
                "gold": 0
            }

        return entity_stats, text_stats

    @staticmethod
    def _get_confusion_matrix_items(gold: dict, predicted: dict):

        gold_set = set(gold.items())
        predicted_set = set(predicted.items())
        # entities which should have been predicted, but are not -> false negatives
        false_negatives = gold_set.difference(predicted_set)
        # entities which are predicted by the model, but miss in the gold set -> false positives
        false_positives = predicted_set.difference(gold_set)
        # entities which are in both sets
        true_positives = predicted_set.intersection(gold_set)

        return false_negatives, false_positives, true_positives

    def _evaluate(self, document, gold_list: list):

        predicted_list = self.spans_to_ent_tuples(document.ents)
        # "hashing" entities into "(start_char, end_char): label" dictionaries
        predicted_dict = self.build_entity_dict(predicted_list)
        gold_dict = self.build_entity_dict(gold_list)

        entity_stats, text_stats = self._initialize_stats_dicts()

        for token in document:
            if not token.is_punct:
                continue
            text_stats["total"] += 1

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
            text_stats[item_name] = len(item)
        text_stats["tn"] = np.max([
            text_stats["total"] - (text_stats["tp"] + text_stats["fp"] + text_stats["fn"]),
            0
        ])

        for stat_key, stat_value in text_stats.items():
            assert stat_value >= 0, f"The value for \"{stat_key}\" is {stat_value}"

        text_overall_scores = self._get_scores(text_stats)
        per_entity_scores = np.array(
            [self._get_scores(entity_stats[entity], include_accuracy=False) for entity in entity_stats.keys()]
        )

        return text_overall_scores, per_entity_scores

    @staticmethod
    def _set_stats_per_entity(entity_stats: dict, confusion_matrix: dict):
        for item_name, items in confusion_matrix.items():
            # items is a set of false/true_positives / false_negatives
            for ent_range, ent_label in items:
                entity_stats[ent_label][item_name] += 1

        return entity_stats

    def _get_scores(self, stats: dict, include_accuracy=True) -> np.array:
        if stats["tp"] + stats["fp"] != 0:
            precision = stats["tp"] / (stats["tp"] + stats["fp"])
        else:
            # Model has found 0 entities -> case is correct only if the model should not have found any annotations
            precision = 1.0 if stats["gold"] == 0 else 0.0

        if stats["tp"] + stats["fn"] != 0:
            recall = stats["tp"] / (stats["tp"] + stats["fn"])
        else:
            # gold entities list contains 0 entities -> 0 should be found
            recall = 1.0 if stats["predicted"] == 0 else 0.0

        f1_score = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0.0

        if include_accuracy:
            if stats["total"] != 0:
                acc = (stats["tp"] + stats["tn"]) / (stats["tp"] + stats["fp"] + stats["tn"] + stats["fn"])
            else:
                # Token contains 0 non-punctuation tokens.
                acc = np.nan
        else:
            acc = np.nan

        if include_accuracy:
            logging.debug("{:.3f} | {:.3f} | {:.3f} | {:.3f}".format(precision, recall, acc, f1_score))
        else:
            logging.debug("{:.3f} | {:.3f} | {} | {:.3f}".format(precision, recall, " " * 5, f1_score))

        result_scores = [precision, recall, acc, f1_score] if include_accuracy else [precision, recall, f1_score]
        for score in result_scores:
            assert 0.0 <= score <= 1.0, f"The \"{score}\" is unbound."

        return np.array(result_scores)

    def _aggregate_and_print_results(self, metrics: np.array, entity_metrics: np.array):
        metrics_aggregation = np.nanmean(metrics, axis=0)
        for idx, metric in enumerate(self._used_metrics):
            logging.debug(f"Setting \"{metric}\"")
            self._used_metrics[metric] = metrics_aggregation[idx]

        means_per_entity = np.nanmean(entity_metrics, axis=0)
        print(f"Average performance per {self._annotated_count} texts:")
        for metric in self._used_metrics:
            print("\t{}: {:.3f}".format(metric, self._used_metrics[metric]))
        for idx, ent_type in enumerate(self._available_entities):
            print(
                f"\t{ent_type}",
                "\t\tPrecision: {:.3f}".format(means_per_entity[idx, 0]),
                "\t\tRecall: {:.3f}".format(means_per_entity[idx, 0]),
                "\t\tF1 Score: {:.3f}".format(means_per_entity[idx, 0]),
                sep="\n"
            )

    """def evaluate_texts(self, texts_to_evaluate: list):
        metrics = []
        entity_class_metrics = []

        txt_idx = 0
        all_texts, all_gold_entities = zip(*texts_to_evaluate)
        # processing the given batch of texts to get the predictions
        for txt_idx, doc in enumerate(self._nlp.pipe(all_texts, batch_size=512)):
            if txt_idx % 100 == 0:
                # reducing the number of output messages
                print(f"Evaluating text {txt_idx + 1} / {len(texts_to_evaluate)}")
            text_metrics, per_entity_metrics = self._evaluate(doc, all_gold_entities[txt_idx]["entities"])
            metrics.append(text_metrics)
            entity_class_metrics.append(per_entity_metrics)

        self._annotated_count = txt_idx + 1
        self._aggregate_and_print_results(np.array(metrics), np.array(entity_class_metrics))
    """


class IntersectionEvaluator(Evaluator):

    def __init__(self, model_to_evaluate: Path):
        super().__init__(model_to_evaluate)

    def _get_scores(self, stats):
        super()._get_scores(stats)

    def _intersection_over_union(self, prediction_range: set, gold_range: set) -> float:
        ent_intersection = prediction_range.intersection(gold_range)
        ent_union = prediction_range.union(gold_range)

        iou = len(ent_intersection) / len(ent_union)
        assert 0 <= iou <= 1, f"IOU equals {iou}"
        return iou

    def _find_entity_match_score(self, prediction: tuple, gold_list: list):
        pred_start, pred_end, pred_label = prediction
        prediction_range = set(range(pred_start, pred_end))
        matching_gold = None
        entity_iou = 0.0
        for gold_start, gold_end, gold_label in gold_list:
            # if gold_end < pred_start:
            #    continue
            # if gold_start > pred_end:
            #    break

            gold_range = set(range(gold_start, gold_end))
            entity_iou = self._intersection_over_union(prediction_range, gold_range)
            if entity_iou > 0:
                matching_gold = (gold_start, gold_end, gold_label)
                break
        if matching_gold is None:
            # no matching gold entity has been found -> false positive prediction
            self._entity_stats[pred_label]["fp"] += 1
            return entity_iou   # 0.0

        match_start, match_end, match_label = matching_gold
        assert entity_iou > 0.0, f"Non overlapping"
        if pred_label == match_label:
            if entity_iou == 1.0:
                self._entity_stats[pred_label]["tp"] += 1
            else:
                self._entity_stats[pred_label]["tp_partial"] += entity_iou
        else:
            if entity_iou == 1.0:
                # predicted label is not correct, but at least the range corresponds
                # self._entity_stats[pred_label]["incorrect"] += 1
                self._entity_stats[pred_label]["tp"] += 0.4
            else:
                self._entity_stats[pred_label]["incorrect"] += 1

        return entity_iou

    def _find_entities_intersections(self, document, gold_list: list) -> np.array:
        entity_ious = []
        predicted_list = self.spans_to_ent_tuples(document.ents)
        for prediction in predicted_list:
            match_iou = self._find_entity_match_score(prediction, gold_list)
            entity_ious.append(match_iou)

        false_negatives = set(gold_list).difference(predicted_list)
        for fn_start, fn_end, fn_label in false_negatives:
            self._entity_stats[fn_label]["fn"] += 1
        return np.array(entity_ious)

    def evaluate_texts(self, texts_to_evaluate: list):
        metrics = []
        entity_metrics = []

        txt_idx = 0
        all_texts, all_gold_entities = zip(*texts_to_evaluate)
        for txt_idx, doc in enumerate(self._nlp.pipe(all_texts, batch_size=512)):
            if txt_idx % 100 == 0:
                print(f"Evaluating text {txt_idx + 1} / {len(texts_to_evaluate)}")

            text_metrics = self._find_entities_intersections(doc, all_gold_entities[txt_idx]["entities"])
            metrics.append(text_metrics)