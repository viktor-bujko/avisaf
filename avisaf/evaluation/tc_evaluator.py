#!/usr/bin/env python3

import logging
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import learning_curve

logger = logging.getLogger(str(__file__))


class ASRSReportClassificationEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def evaluate(predictions: list, test_target):

        ensemble = ""
        if len(predictions) > 1:
            logger.debug(f"{len(predictions)} models ensembling")
            ensemble = f"(ensemble of {len(predictions)} models)"

        predictions = np.mean(predictions, axis=0)

        for predictions_distribution, class_targets in zip(predictions, test_target):
            class_predictions = np.argmax(predictions_distribution, axis=1)
            unique_predictions_count = np.unique(class_targets).shape[0]
            avg = "binary" if unique_predictions_count == 2 else "macro"

            print("==============================================")
            print(
                "Confusion matrix: number [i,j] indicates the number of observations of class i which were predicted to be in class j"
            )
            print(metrics.confusion_matrix(class_targets, class_predictions))
            if ensemble:
                print(ensemble)
            print(
                "Model Based Accuracy: {:.2f}".format(
                    metrics.accuracy_score(class_targets, class_predictions) * 100
                )
            )
            print(
                "Model Based Balanced Accuracy: {:.2f}".format(
                    metrics.balanced_accuracy_score(class_targets, class_predictions)
                    * 100
                )
            )
            print(
                "Model Based ROC-AUC: {:.2f}".format(
                    metrics.roc_auc_score(
                        class_targets, predictions_distribution, multi_class="ovr"
                    )
                    * 100
                )
            )

            print(
                "Model Based Macro Precision: {:.2f}".format(
                    metrics.precision_score(
                        class_targets, class_predictions, average=avg
                    )
                    * 100
                )
            )
            print(
                "Model Based Macro Recall: {:.2f}".format(
                    metrics.recall_score(class_targets, class_predictions, average=avg)
                    * 100
                )
            )
            print(
                "Model Based Macro F1-score: {:.2f}".format(
                    metrics.f1_score(class_targets, class_predictions, average=avg)
                    * 100
                )
            )
            print("==============================================")
            for unique_prediction in range(unique_predictions_count):
                mockup_predictions = np.full(class_targets.shape, unique_prediction)
                print(
                    f"Accuracy predicting always {unique_prediction}: {metrics.accuracy_score(class_targets, mockup_predictions) * 100}"
                )
                print(
                    f"Accuracy predicting always {unique_prediction}: {metrics.balanced_accuracy_score(class_targets, mockup_predictions) * 100}"
                )
                print(
                    f"F1-score: {metrics.f1_score(class_targets, mockup_predictions, average=avg) * 100}"
                )
                print(
                    f"Model Based Precision: {metrics.precision_score(class_targets, mockup_predictions, zero_division=1, average=avg) * 100}"
                )
                print(
                    f"Model Based Recall: {metrics.recall_score(class_targets, mockup_predictions, average=avg) * 100}"
                )
                print("==============================================")

    @staticmethod
    def plot(probability_predictions, test_target):
        preds = np.mean(probability_predictions, axis=0)[:, 1]

        fpr, tpr, threshold = metrics.roc_curve(test_target, preds)
        roc_auc = metrics.auc(fpr, tpr)
        # prec, recall, thr = metrics.precision_recall_curve(test_target, preds)

        plt.title("ROC Curve")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()

    @staticmethod
    def plot_learning_curve(
        estimator,
        title,
        x,
        y,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
    ):
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            x,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, "o-")
        axes[2].fill_between(
            fit_times_mean,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt
