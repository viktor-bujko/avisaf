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
        self._visualizer = Visualizer()

    def evaluate_dummy_baseline(self, target_classes: np.ndarray):
        unique_targets_count = np.unique(target_classes).shape[0]
        for baseline_mockup in range(unique_targets_count):
            mockup_predictions = np.zeros((target_classes.shape[0], unique_targets_count))
            mockup_predictions[:, baseline_mockup] = 1
            baseline_conf_matrix, baseline_results_dict = self.evaluate(mockup_predictions, target_classes, show_curves=False, model_type="dummy_" + str(baseline_mockup))
            self._visualizer.print_metrics("Baseline metrics: ", baseline_conf_matrix, baseline_results_dict)

    def evaluate_random_predictions(self, target_classes: np.ndarray, show_curves: bool = False):
        unique_targets_count = np.unique(target_classes).shape[0]
        random_idxs = np.random.randint(0, unique_targets_count, size=target_classes.shape[0])
        random_predictions = np.zeros((target_classes.shape[0], unique_targets_count))
        for idx, x in enumerate(random_idxs):
            random_predictions[idx, x] = 1
        random_conf_matrix, random_results_dict = self.evaluate(random_predictions, target_classes, show_curves=show_curves, model_type="random predictions")
        self._visualizer.print_metrics("Random metrics: ", random_conf_matrix, random_results_dict)

    def evaluate(self, predictions_distribution: np.ndarray, target_classes: np.ndarray, avg_method: [str, None] = None, show_curves: bool = False, model_type: str = "prediction model") -> tuple:
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

        if show_curves:
            roc_curves_data, prec_recall_data = [], []
            class_predictions = np.argmax(predictions_distribution, axis=1)
            add_string = (f" for \"{self._evaluated_topic_label}\" classes" if self._evaluated_topic_label else "") + f" ({model_type})"
            roc_curves_title = "ROC Curve" + add_string
            prec_recall_title = "Precision-recall curve" + add_string
            precision = metrics.precision_score(target_classes, class_predictions, average=avg_method)
            for positive_class in np.unique(target_classes):
                # inverse_transforms returns the list of decoded labels - list now contains only 1 item
                label = self._label_encoder.inverse_transform([positive_class])[0]
                fpr, tpr, thresholds = metrics.roc_curve(target_classes, predictions_distribution[:, positive_class], pos_label=positive_class)
                roc_curves_data.append((fpr, tpr, thresholds, label))
                prec, recall, thresholds = metrics.precision_recall_curve(target_classes, predictions_distribution[:, positive_class], pos_label=positive_class)
                prec_recall_data.append((prec, recall, precision[positive_class], label))

            self._visualizer.plot_evaluation_curves(
                prec_recall_data,
                title=prec_recall_title,
                xlabel="Recall",
                ylabel="Precision",
                label="AP",
                model_type=model_type
            )
            self._visualizer.plot_evaluation_curves(
                roc_curves_data,
                label_method=metrics.auc,
                title=roc_curves_title,
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                label="AUC",
                show_diagonal=True,
                model_type=model_type
            )

        return confusion_matrix, results
