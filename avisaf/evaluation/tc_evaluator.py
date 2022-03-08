#!/usr/bin/env python3

import logging
import numpy as np
import sklearn.metrics as metrics
from evaluation.visualizer import Visualizer

logger = logging.getLogger("avisaf_logger")


class ASRSReportClassificationEvaluator:
    def __init__(self, evaluated_label: str = None, label_encoder=None):
        self._evaluated_topic_label = evaluated_label
        self._label_encoder = label_encoder
        self._visualizer = Visualizer(evaluated_label, label_encoder)

    def evaluate_dummy_baseline(self, target_classes: np.ndarray):
        unique_targets_count = np.unique(target_classes).shape[0]
        for baseline_mockup in range(unique_targets_count):
            mockup_predictions = np.zeros((target_classes.shape[0], unique_targets_count))
            mockup_predictions[:, baseline_mockup] = 1
            baseline_conf_matrix, baseline_results_dict = self.evaluate(mockup_predictions, target_classes)
            # self._visualizer.show_curves(mockup_predictions, target_classes, model_type="dummy_" + str(baseline_mockup), avg_method=None)
            self._visualizer.print_metrics("Baseline metrics: ", baseline_conf_matrix, baseline_results_dict)

    def evaluate_random_predictions(self, target_classes: np.ndarray, show_curves: bool = False):
        unique_targets_count = np.unique(target_classes).shape[0]
        random_idxs = np.random.randint(0, unique_targets_count, size=target_classes.shape[0])
        random_predictions = np.zeros((target_classes.shape[0], unique_targets_count))
        for idx, x in enumerate(random_idxs):
            random_predictions[idx, x] = 1
        random_conf_matrix, random_results_dict = self.evaluate(random_predictions, target_classes)
        if show_curves:
            self._visualizer.show_curves(random_predictions, target_classes, model_type="random predictions", avg_method=None)
        self._visualizer.print_metrics("Random metrics: ", random_conf_matrix, random_results_dict)

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
