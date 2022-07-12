#!/usr/bin/env python3

import sys
import matplotlib
import logging
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from pathlib import Path

logger = logging.getLogger("avisaf_logger")


class Visualizer:

    def __init__(self, model_dir: str = None):
        self._model_dir = model_dir
        self._open_files = {}

    def print_metrics(self, title: str, classes: list, model_conf_matrix, model_results_dict: dict, filename: str):

        stdout = sys.stdout
        files_to_write = {stdout}  # results will always be written to stdout
        if self._model_dir is not None:
            if self._open_files.get(filename, 0) == 0:
                # the file is accessed first time in this program run - rewriting the content
                access = "w"
                self._open_files[filename] = 1
            else:
                # accessing the same file multiple times during program run
                access = "a"
                assert self._open_files[filename] >= 1
                self._open_files[filename] += 1
            file_stream = open(Path(self._model_dir,  f"{filename}.txt"), access)
            files_to_write.add(file_stream)

        for f in files_to_write:
            sys.stdout = f
            print(title)
            print(f"Classes order: { ' | '.join(classes)}")
            print(model_conf_matrix)
            for metric_name, value in model_results_dict.items():
                if isinstance(value, float):
                    print(f"\t{metric_name}: %0.2f" % (value * 100))
                elif isinstance(value, list) or isinstance(value, np.ndarray):
                    formatted_floats_list = [("| %0.2f |" % (fl_number * 100)) for fl_number in value]
                    print(f"\t{metric_name}: {''.join(formatted_floats_list)}")
                    print(f"\tAverage {metric_name}: {'%0.2f' % (np.mean(value) * 100)}")
                    print(f"\tStd dev of {metric_name}: {'%0.2f' % (np.std(value) * 100)}")
                else:
                    print(f"\t{metric_name}: {value}")
            if f != stdout:
                f.close()
        sys.stdout = stdout

    def show_curves(
            self,
            predictions_distribution: np.ndarray,
            target_classes: np.ndarray,
            model_type: str,
            topic_label: str = None,
            label_encoder=None
    ):
        add_string = (f" for \"{topic_label}\" classes" if topic_label else "") + f" ({model_type})"
        fname = f"{'_' + topic_label if topic_label else ''}_{model_type}.svg"

        self.compute_curve(
            "ROC Curve" + add_string,
            target_classes,
            label_encoder,
            predictions_distribution,
            method=metrics.roc_curve,
            xlabel="False Positive Rate",
            ylabel="True Positive Rate (Recall)",
            savefig_fname="roc_curve" + fname
        )

        self.compute_curve(
            "Precision-recall curve" + add_string,
            target_classes,
            label_encoder,
            predictions_distribution,
            method=metrics.precision_recall_curve,
            xlabel="Recall",
            ylabel="Precision",
            savefig_fname="precision_recall" + fname
        )

    def compute_curve(self, title, target_classes, encoder, pred_dist, method, savefig_fname, **kwargs):
        fprs, tprs, thresholds = [], [], []
        plt.title(title)
        for positive_class in np.unique(target_classes):
            # inverse_transforms returns the list of decoded labels - contains 1 item only
            label = encoder.inverse_transform([positive_class])[0]
            fpr, tpr, threshold = method(
                target_classes, pred_dist[:, positive_class],
                pos_label=positive_class
            )
            self.plot_evaluation_curves(
                fpr, tpr,
                label=f"AUC ({label}): %0.2f" % metrics.auc(fpr, tpr),
                xlabel=kwargs.get("xlabel", ""),
                ylabel=kwargs.get("ylabel", ""),
                show_diag=True
            )
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)

        self.save_plot(savefig_fname)
        self.show()

    def save_plot(self, filename: str):
        model_dir = self._model_dir if self._model_dir else "../evaluation"
        fname = Path(model_dir, filename)

        logger.info(f"Saving figure to: {fname}")
        plt.savefig(fname=fname)

    @staticmethod
    def show():
        non_gui_backend = matplotlib.get_backend() in ['agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg']
        if non_gui_backend:
            logger.warning(f"Non GUI backend is active. Saved the figure.")
            plt.clf()
            return

        plt.show()

    @staticmethod
    def plot_evaluation_curves(data_x, data_y, **kwargs):

        plt.plot(data_x, data_y, label=kwargs.get("label", ""))
        plt.legend(loc="best")
        plt.grid(visible=True, linestyle='--', alpha=0.4)
        if kwargs.get("show_diag", False):
            # plotting random classifier roc
            plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel(kwargs.get("ylabel", ""))
        plt.xlabel(kwargs.get("xlabel", ""))
        plt.xticks(np.arange(0, 1.05, 0.1))
        plt.yticks(np.arange(0, 1.05, 0.1))
        plt.tight_layout()
