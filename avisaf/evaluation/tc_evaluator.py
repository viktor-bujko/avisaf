#!/usr/bin/env python3
import json
import lzma
import pickle
from pathlib import Path

import numpy as np
import sklearn.metrics as metrics

from main import logger
from classification.predictor_decoder import ASRSReportClassificationPredictor
from evaluation.visualizer import Visualizer
from util.data_extractor import CsvAsrsDataExtractor


class ASRSReportClassificationEvaluator:
    def __init__(self, evaluated_label: str = None, label_encoder=None, model_dir: str = None):
        self._evaluated_topic_label = evaluated_label
        self._label_encoder = label_encoder
        self._visualizer = Visualizer(evaluated_label, label_encoder, model_dir)

    def evaluate_dummy_baseline(self, target_classes: np.ndarray):
        unique_targets_count = np.unique(target_classes).shape[0]
        for baseline_mockup in range(unique_targets_count):
            mockup_predictions = np.zeros((target_classes.shape[0], unique_targets_count))
            mockup_predictions[:, baseline_mockup] = 1
            baseline_conf_matrix, baseline_results_dict = self.evaluate(mockup_predictions, target_classes)
            # self._visualizer.show_curves(mockup_predictions, target_classes, model_type="dummy_" + str(baseline_mockup), avg_method=None)
            self._visualizer.print_metrics(f"Baseline {self._evaluated_topic_label if self._evaluated_topic_label else ''} metrics: ", baseline_conf_matrix, baseline_results_dict, "results_dummy")

    def evaluate_random_predictions(self, target_classes: np.ndarray, show_curves: bool = False):
        unique_targets_count = np.unique(target_classes).shape[0]
        random_idxs = np.random.randint(0, unique_targets_count, size=target_classes.shape[0])
        random_predictions = np.zeros((target_classes.shape[0], unique_targets_count))
        for idx, x in enumerate(random_idxs):
            random_predictions[idx, x] = 1
        random_conf_matrix, random_results_dict = self.evaluate(random_predictions, target_classes)
        if show_curves:
            self._visualizer.show_curves(random_predictions, target_classes, model_type="random_predictions", avg_method=None)
        self._visualizer.print_metrics("Random metrics: ", random_conf_matrix, random_results_dict, "results_random")

    def evaluate(self, predictions_distribution: np.ndarray, target_classes: np.ndarray, avg_method: [str, None] = None) -> tuple:
        class_predictions = np.argmax(predictions_distribution, axis=1)
        confusion_matrix = metrics.confusion_matrix(target_classes, class_predictions)
        results = {
            "Accuracy": metrics.accuracy_score(target_classes, class_predictions),
            "Precision": metrics.precision_score(target_classes, class_predictions, average=avg_method, zero_division=0),
            "Recall": metrics.recall_score(target_classes, class_predictions, average=avg_method, zero_division=0),
            "F1 score": metrics.f1_score(target_classes, class_predictions, average=avg_method, zero_division=0),
            "ROC AUC macro ovr": metrics.roc_auc_score(target_classes, predictions_distribution, multi_class="ovr", average="macro"),
            "ROC AUC weighted ovr": metrics.roc_auc_score(target_classes, predictions_distribution, multi_class="ovr", average="weighted")
        }

        return confusion_matrix, results


def evaluate_classification(model_path: str, text_paths: list, show_curves: bool, compare_baseline: bool):

    if not model_path or not text_paths:
        logger.error("Both model_path and text_path arguments must be specified")
        return

    with lzma.open(Path(model_path, "classifier.model"), "rb") as model_file,\
         open(Path(model_path, "parameters.json"), "r") as model_parameters:
        model_predictors, label_encoders = pickle.load(model_file)
        model_parameters = json.load(model_parameters)

    extractor = CsvAsrsDataExtractor(text_paths)
    predictor = ASRSReportClassificationPredictor(extractor, model_parameters.get("vectorizer_params", {}).get("vectorizer"))

    predictions_targets = predictor.get_evaluation_predictions(model_predictors, model_parameters.get("trained_labels"))
    for (predictions, targets), topic_label, label_encoder in zip(predictions_targets, model_predictors.keys(), label_encoders):
        evaluator = ASRSReportClassificationEvaluator(topic_label, label_encoder, model_path)
        model_conf_matrix, model_results_dict = evaluator.evaluate(predictions, targets)
        visualizer = Visualizer(topic_label, label_encoder, model_path)
        visualizer.print_metrics(f"Evaluating '{topic_label}' predictor:", model_conf_matrix, model_results_dict, "results_eval")
        if show_curves:
            visualizer.show_curves(predictions, targets, avg_method=None)
        if compare_baseline:
            # generate baseline predictions and evaluate them
            evaluator.evaluate_dummy_baseline(targets)
            evaluator.evaluate_random_predictions(targets, show_curves=show_curves)