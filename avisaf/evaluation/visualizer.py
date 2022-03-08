#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


class Visualizer:

    @staticmethod
    def print_metrics(title: str, model_conf_matrix, model_results_dict: dict):

        print(title)
        print(model_conf_matrix)
        for metric_name, value in model_results_dict.items():
            if isinstance(value, float):
                print(f"\t{metric_name}: %0.2f" % (value * 100))
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                formatted_floats_list = [("| %0.2f |" % fl_number) for fl_number in value]
                print(f"\t{metric_name}: {''.join(formatted_floats_list)}")
            else:
                print(f"\t{metric_name}: {value}")

    @staticmethod
    def plot_evaluation_curves(curves_data: list, **kwargs):

        plt.title(kwargs.get("title", ""))
        for fpr, tpr, value, predicted_class_name in curves_data:
            if kwargs.get("label_method"):
                value = kwargs.get("label_method")(fpr, tpr)
            plt.plot(fpr, tpr, label=kwargs.get("label", "") + (f" ({predicted_class_name}) = %0.2f" % value))
        if kwargs.get("show_diagonal", False):
            plt.plot([0, 1], [0, 1], "r--")
        label_plot = plt.subplot()
        box = label_plot.get_position()
        # label_plot.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        label_plot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        label_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        # plt.legend(loc="lower right")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel(kwargs.get("ylabel", ""))
        plt.xlabel(kwargs.get("xlabel", ""))
        plt.tight_layout()
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
        axes[0].plot_evaluation_curves(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        axes[0].plot_evaluation_curves(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot_evaluation_curves(train_sizes, fit_times_mean, "o-")
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
        axes[2].plot_evaluation_curves(fit_times_mean, test_scores_mean, "o-")
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
